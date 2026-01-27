#include <mpi.h>        // MPI функции (Init, Scatterv, Gatherv, Wtime и т.д.)
#include <iostream>     // cout/cerr
#include <vector>       // std::vector
#include <random>       // генерация случайных чисел
#include <iomanip>      // setprecision
#include <cmath>        // abs

/*
  Пример локальной обработки массива (одинаково на каждом процессе):
  y[i] = a * x[i] + b

  Важно:
  - Логика должна быть одинаковой на всех процессах
  - Каждый процесс обрабатывает только свой кусок
*/
static void process_chunk(const float* x, float* y, int n, float a, float b)
{
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + b;
    }
}

int main(int argc, char** argv)
{
    // 1) Инициализация MPI (обязательный старт)
    MPI_Init(&argc, &argv);

    // 2) Узнаём номер процесса (rank) и общее количество процессов (size)
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 3) Общий размер массива (можно менять, но для замеров лучше побольше)
    const int N = 10'000'000;

    // 4) Параметры "обработки"
    const float a = 1.7f;
    const float b = 0.3f;

    /*
      5) Разбиение массива по процессам
      counts[p]  — сколько элементов получит процесс p
      displs[p]  — с какого индекса в общем массиве начинается его кусок

      Пример: N=10, size=3
      base=3, rem=1
      counts = [4,3,3]
      displs = [0,4,7]
    */
    std::vector<int> counts(size);
    std::vector<int> displs(size);

    int base = N / size;     // сколько элементов минимум на каждый процесс
    int rem  = N % size;     // сколько "лишних" элементов надо раздать первым процессам

    int offset = 0;
    for (int p = 0; p < size; ++p) {
        counts[p] = base + (p < rem ? 1 : 0); // первые rem процессов получают на 1 элемент больше
        displs[p] = offset;                   // стартовый индекс
        offset += counts[p];                  // сдвигаем offset
    }

    /*
      6) Данные на корневом процессе rank=0
      Только root хранит полный массив x и полный результат y.
      Остальные процессы хранят только свои куски.
    */
    std::vector<float> x;  // полный вход
    std::vector<float> y;  // полный выход

    if (rank == 0) {
        // Выделяем память под вход и выход на root
        x.resize(N);
        y.resize(N);

        // Заполняем вход случайными числами
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < N; ++i) {
            x[i] = dist(rng);
        }
    }

    // 7) Локальные массивы на каждом процессе (для его куска)
    std::vector<float> x_local(counts[rank]);
    std::vector<float> y_local(counts[rank]);

    /*
      8) Замер времени (MPI_Wtime)
      Чтобы время было честным, синхронизируем процессы барьером.
    */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /*
      9) Scatterv — раздача разных по размеру кусков массива x
      - root отправляет разные куски (counts/displs)
      - каждый процесс получает свой кусок в x_local
    */
    MPI_Scatterv(
        rank == 0 ? x.data() : nullptr,  // буфер отправки (только у root)
        counts.data(),                    // сколько отправлять каждому
        displs.data(),                    // смещения для каждого
        MPI_FLOAT,                        // тип данных
        x_local.data(),                   // куда принимать на каждом процессе
        counts[rank],                     // сколько принимать этому процессу
        MPI_FLOAT,                        // тип данных
        0,                                // root = 0
        MPI_COMM_WORLD
    );

    // 10) Локальная обработка на каждом процессе
    process_chunk(x_local.data(), y_local.data(), counts[rank], a, b);

    /*
      11) Gatherv — сбор результата обратно на root
      - каждый процесс отправляет свой y_local
      - root собирает всё в y
    */
    MPI_Gatherv(
        y_local.data(),                   // отправляемый буфер (локальный результат)
        counts[rank],                     // сколько элементов отправляем
        MPI_FLOAT,                        // тип
        rank == 0 ? y.data() : nullptr,    // root принимает в полный массив y
        counts.data(),                    // сколько принять от каждого
        displs.data(),                    // смещения для каждого
        MPI_FLOAT,                        // тип
        0,                                // root = 0
        MPI_COMM_WORLD
    );

    // 12) Конец замера времени
    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    /*
      13) Вывод результата только на root
      Также делаем небольшую проверку корректности на нескольких индексах.
    */
    if (rank == 0) {
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "Task 4 — MPI distributed array processing\n";
        std::cout << "N = " << N << ", processes = " << size << "\n";
        std::cout << "Elapsed time (s) = " << (t1 - t0) << "\n";

        // Мини-проверка правильности на нескольких точках
        double max_abs_diff = 0.0;

        int test_idx[] = {0, 1, 2, N/2, N-2, N-1};
        for (int idx : test_idx) {
            double ref = (double)a * (double)x[idx] + (double)b;
            double d   = std::abs(ref - (double)y[idx]);
            if (d > max_abs_diff) max_abs_diff = d;
        }

        std::cout << "Max abs diff (sample points) = " << max_abs_diff << "\n";
    }

    // 14) Завершение MPI (обязательное)
    MPI_Finalize();
    return 0;
}
