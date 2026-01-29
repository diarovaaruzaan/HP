#include <mpi.h>        // Основная библиотека MPI
#include <iostream>    // Ввод/вывод
#include <vector>      // Контейнер vector
#include <random>      // Генерация случайных чисел
#include <cmath>       // Математические функции (sqrt)
#include <numeric>    // Дополнительные числовые алгоритмы

// ------------------------------------------------------------
// Функция вычисляет, сколько элементов массива
// получит каждый MPI-процесс (counts)
// и с какого смещения начинать (displs)
// Используется для MPI_Scatterv
// ------------------------------------------------------------
static void build_counts_displs(int N, int size,
                                std::vector<int>& counts,
                                std::vector<int>& displs)
{
    // Инициализация векторов
    counts.assign(size, 0);
    displs.assign(size, 0);

    // Базовое количество элементов на процесс
    int base = N / size;

    // Остаток, который распределяется по первым процессам
    int rem  = N % size;

    int offset = 0;
    for (int r = 0; r < size; ++r) {
        // Первые rem процессов получают на 1 элемент больше
        counts[r] = base + (r < rem ? 1 : 0);

        // Смещение для текущего процесса
        displs[r] = offset;

        // Обновляем смещение
        offset   += counts[r];
    }
}

int main(int argc, char** argv)
{
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    // rank — номер процесса, size — общее число процессов
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер массива (можно передать через аргументы командной строки)
    // Пример: mpirun -np 4 ./task1 1000000
    int N = 1'000'000;
    if (argc >= 2)
        N = std::max(1, std::atoi(argv[1]));

    // Замер времени начала выполнения
    double t0 = MPI_Wtime();

    // ------------------------------------------------------------
    // 1) Процесс rank 0 создаёт исходный массив
    // ------------------------------------------------------------
    std::vector<double> x;
    if (rank == 0) {
        x.resize(N);

        // Генератор случайных чисел
        std::mt19937_64 rng(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Заполнение массива случайными значениями
        for (int i = 0; i < N; ++i)
            x[i] = dist(rng);
    }

    // ------------------------------------------------------------
    // 2) Распределение массива между процессами (MPI_Scatterv)
    // ------------------------------------------------------------
    std::vector<int> counts, displs;
    build_counts_displs(N, size, counts, displs);

    // Количество элементов для текущего процесса
    int local_n = counts[rank];

    // Локальный буфер
    std::vector<double> local(local_n);

    // Распределяем части массива по процессам
    MPI_Scatterv(
        rank == 0 ? x.data() : nullptr, // исходный массив (только у rank 0)
        counts.data(),                  // сколько элементов отправлять
        displs.data(),                  // смещения
        MPI_DOUBLE,                     // тип данных
        local.data(),                   // локальный буфер
        local_n,                        // размер локального буфера
        MPI_DOUBLE,
        0,                              // корневой процесс
        MPI_COMM_WORLD
    );

    // ------------------------------------------------------------
    // 3) Локальные вычисления: сумма и сумма квадратов
    // ------------------------------------------------------------
    double local_sum = 0.0;
    double local_sumsq = 0.0;

    for (int i = 0; i < local_n; ++i) {
        local_sum   += local[i];
        local_sumsq += local[i] * local[i];
    }

    // ------------------------------------------------------------
    // 4) Сбор результатов на rank 0 (MPI_Reduce)
    // ------------------------------------------------------------
    double sum = 0.0;
    double sumsq = 0.0;

    MPI_Reduce(&local_sum, &sum, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&local_sumsq, &sumsq, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // 5) Вычисление среднего и стандартного отклонения
    // ------------------------------------------------------------
    if (rank == 0) {
        // Среднее значение
        double mean = sum / (double)N;

        // Дисперсия
        double var = (sumsq / (double)N) - mean * mean;

        // Защита от отрицательных значений из-за погрешностей
        if (var < 0.0) var = 0.0;

        // Стандартное отклонение
        double stddev = std::sqrt(var);

        // Замер времени окончания
        double t1 = MPI_Wtime();

        // Вывод результатов
        std::cout << "N = " << N << "\n";
        std::cout << "Mean   = " << mean << "\n";
        std::cout << "Stddev = " << stddev << "\n";
        std::cout << "Execution time: "
                  << (t1 - t0) << " seconds\n";
    }

    // Завершение работы MPI
    MPI_Finalize();
    return 0;
}
