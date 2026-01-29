#include <mpi.h>        // MPI функции и типы
#include <iostream>    // вывод в консоль
#include <vector>      // std::vector
#include <random>      // генерация случайных чисел
#include <algorithm>  // std::max
#include <cmath>       // мат. функции (на будущее)

// ------------------------------------------------------------
// Функция распределения строк между MPI-процессами
// rowCounts[r]  — сколько строк получает процесс r
// rowDispls[r]  — с какой глобальной строки начинается процесс r
// ------------------------------------------------------------
static void build_counts_displs_rows(int N, int size,
                                     std::vector<int>& rowCounts,
                                     std::vector<int>& rowDispls)
{
    // Инициализируем массивы распределения
    rowCounts.assign(size, 0);
    rowDispls.assign(size, 0);

    // Сколько строк минимум получает каждый процесс
    int base = N / size;

    // Остаток, который распределяем по первым процессам
    int rem  = N % size;

    int off = 0;
    for (int r = 0; r < size; ++r) {
        // Первые rem процессов получают на одну строку больше
        rowCounts[r] = base + (r < rem ? 1 : 0);

        // Смещение (глобальный индекс первой строки) для процесса r
        rowDispls[r] = off;

        // Увеличиваем смещение
        off += rowCounts[r];
    }
}

int main(int argc, char** argv)
{
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    // rank — номер процесса, size — количество процессов
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Размер графа (N x N)
    // Можно передать аргументом: mpirun -np 4 ./task3 200
    int N = 200;
    if (argc >= 2)
        N = std::max(1, std::atoi(argv[1]));

    // Большое число для "бесконечности" (нет пути)
    const double INF = 1e18;

    // Начало замера времени
    double t0 = MPI_Wtime();

    // ------------------------------------------------------------
    // 1) Процесс rank 0 создаёт матрицу смежности графа (весов)
    // ------------------------------------------------------------
    std::vector<double> G;
    if (rank == 0) {
        // Изначально все расстояния INF
        G.assign((size_t)N * N, INF);

        // Генераторы случайных чисел
        std::mt19937_64 rng(123);

        // веса ребёр: 1..10
        std::uniform_real_distribution<double> wdist(1.0, 10.0);

        // вероятность ребра: 0..1
        std::uniform_real_distribution<double> pdist(0.0, 1.0);

        // Строим случайный ориентированный граф
        for (int i = 0; i < N; ++i) {
            // расстояние до себя = 0
            G[(size_t)i * N + i] = 0.0;

            for (int j = 0; j < N; ++j) {
                if (i == j) continue;

                // С вероятностью 0.2 добавляем ребро i -> j
                if (pdist(rng) < 0.2) {
                    G[(size_t)i * N + j] = wdist(rng);
                }
            }
        }
    }

    // ------------------------------------------------------------
    // 2) Распределяем строки матрицы между процессами
    // ------------------------------------------------------------
    std::vector<int> rowCounts, rowDispls;
    build_counts_displs_rows(N, size, rowCounts, rowDispls);

    int localRows = rowCounts[rank]; // сколько строк у текущего процесса

    // Для Scatterv/Allgatherv нужны counts/displs в элементах (rows * N)
    std::vector<int> sendCounts(size), sendDispls(size);
    for (int r = 0; r < size; ++r) {
        sendCounts[r] = rowCounts[r] * N;
        sendDispls[r] = rowDispls[r] * N;
    }

    // Локальный кусок матрицы расстояний (только свои строки)
    std::vector<double> localD((size_t)localRows * N, INF);

    // Рассылаем строки матрицы от rank 0 ко всем процессам
    MPI_Scatterv(rank == 0 ? G.data() : nullptr,   // источник (только rank 0)
                 sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                 localD.data(), localRows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // 3) globalD хранится на каждом процессе
    // Нужно, чтобы иметь доступ к строке k (rowK) на каждой итерации
    // ------------------------------------------------------------
    std::vector<double> globalD((size_t)N * N, INF);

    // Собираем общий массив globalD на всех процессах
    MPI_Allgatherv(localD.data(), localRows * N, MPI_DOUBLE,
                   globalD.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // 4) Алгоритм Флойда–Уоршелла
    // Основная идея:
    // dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
    // ------------------------------------------------------------
    for (int k = 0; k < N; ++k) {

        // Получаем указатель на строку k из globalD
        const double* rowK = &globalD[(size_t)k * N];

        // Каждый процесс обновляет только свои строки i (localD)
        for (int li = 0; li < localRows; ++li) {

            // Глобальный индекс строки i
            int gi = rowDispls[rank] + li;

            // Указатель на строку i в локальной матрице
            double* rowI = &localD[(size_t)li * N];

            // dist(i,k)
            double dik = rowI[k];

            // Если пути до k нет — пропускаем
            if (dik >= INF / 2) continue;

            // Обновляем все j
            for (int j = 0; j < N; ++j) {
                double alt = dik + rowK[j];   // путь через k
                if (alt < rowI[j]) rowI[j] = alt;
            }
        }

        // ------------------------------------------------------------
        // 5) После обновления своих строк
        // делаем обмен между процессами
        // чтобы на следующей итерации k+1
        // все процессы имели актуальную globalD
        // ------------------------------------------------------------
        MPI_Allgatherv(localD.data(), localRows * N, MPI_DOUBLE,
                       globalD.data(), sendCounts.data(), sendDispls.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    // ------------------------------------------------------------
    // 6) Вывод результатов только на rank 0
    // ------------------------------------------------------------
    if (rank == 0) {
        double t1 = MPI_Wtime();

        std::cout << "N = " << N << "\n";
        std::cout << "dist[0][0] = " << globalD[0] << "\n";
        std::cout << "dist[0][N-1] = " << globalD[N - 1] << "\n";
        std::cout << "dist[N-1][0] = "
                  << globalD[(size_t)(N - 1) * N + 0] << "\n";

        std::cout << "Execution time: " << (t1 - t0) << " seconds\n";
    }

    // Завершение MPI
    MPI_Finalize();
    return 0;
}

