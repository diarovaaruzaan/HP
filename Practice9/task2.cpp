#include <mpi.h>        // Основная библиотека MPI
#include <iostream>    // Ввод / вывод
#include <vector>      // Контейнер vector
#include <random>      // Генерация случайных чисел
#include <cmath>       // Математические функции (fabs)
#include <algorithm>  // std::max

// ------------------------------------------------------------
// Функция распределяет строки матрицы между процессами
// rowCounts[r]  — сколько строк получает процесс r
// rowDispls[r]  — с какой глобальной строки начинается процесс r
// Используется для MPI_Scatterv / MPI_Gatherv
// ------------------------------------------------------------
static void build_counts_displs_rows(int N, int size,
                                     std::vector<int>& rowCounts,
                                     std::vector<int>& rowDispls)
{
    rowCounts.assign(size, 0);
    rowDispls.assign(size, 0);

    int base = N / size;   // базовое число строк на процесс
    int rem  = N % size;   // остаток строк

    int off = 0;
    for (int r = 0; r < size; ++r) {
        // первые rem процессов получают на одну строку больше
        rowCounts[r] = base + (r < rem ? 1 : 0);
        rowDispls[r] = off;
        off += rowCounts[r];
    }
}

// ------------------------------------------------------------
// Определяет, какой MPI-процесс владеет глобальной строкой k
// Нужно для выбора владельца ведущей строки (pivot)
// ------------------------------------------------------------
static int owner_of_row(int k,
                        const std::vector<int>& rowCounts,
                        const std::vector<int>& rowDispls)
{
    int size = (int)rowCounts.size();
    for (int r = 0; r < size; ++r) {
        int start = rowDispls[r];
        int end   = start + rowCounts[r];
        if (k >= start && k < end)
            return r;
    }
    return 0;
}

int main(int argc, char** argv)
{
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // номер процесса
    MPI_Comm_size(MPI_COMM_WORLD, &size); // число процессов

    // Размер системы (можно передать аргументом)
    // Пример: mpirun -np 4 ./task2 256
    int N = 256;
    if (argc >= 2)
        N = std::max(1, std::atoi(argv[1]));

    // Начало замера времени
    double t0 = MPI_Wtime();

    // ------------------------------------------------------------
    // 1) Процесс rank 0 создаёт матрицу A и вектор b
    // ------------------------------------------------------------
    std::vector<double> A, b;
    if (rank == 0) {
        A.assign((size_t)N * N, 0.0);
        b.assign(N, 0.0);

        std::mt19937_64 rng(123);
        std::uniform_real_distribution<double> dist(0.0, 1.0);

        // Генерация диагонально доминируемой матрицы
        for (int i = 0; i < N; ++i) {
            double rowsum = 0.0;
            for (int j = 0; j < N; ++j) {
                if (i == j) continue;
                double v = dist(rng);
                A[(size_t)i * N + j] = v;
                rowsum += std::fabs(v);
            }
            // Диагональный элемент больше суммы остальных
            A[(size_t)i * N + i] = rowsum + 1.0;
            b[i] = dist(rng);
        }
    }

    // ------------------------------------------------------------
    // 2) Распределение строк матрицы между процессами
    // ------------------------------------------------------------
    std::vector<int> rowCounts, rowDispls;
    build_counts_displs_rows(N, size, rowCounts, rowDispls);

    int localRows = rowCounts[rank]; // сколько строк у текущего процесса

    // Для матрицы считаем количество элементов (rows * N)
    std::vector<int> sendCountsA(size), sendDisplsA(size);
    for (int r = 0; r < size; ++r) {
        sendCountsA[r] = rowCounts[r] * N;
        sendDisplsA[r] = rowDispls[r] * N;
    }

    // Локальные части матрицы и вектора
    std::vector<double> localA((size_t)localRows * N);
    std::vector<double> localb(localRows);

    // Распределение строк матрицы A
    MPI_Scatterv(rank == 0 ? A.data() : nullptr,
                 sendCountsA.data(), sendDisplsA.data(), MPI_DOUBLE,
                 localA.data(), localRows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Распределение элементов вектора b
    MPI_Scatterv(rank == 0 ? b.data() : nullptr,
                 rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                 localb.data(), localRows, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // 3) Прямой ход метода Гаусса (с MPI_Bcast)
    // ------------------------------------------------------------

    // Буфер для ведущей строки (pivot): N коэффициентов + b
    std::vector<double> pivotRow(N + 1, 0.0);

    for (int k = 0; k < N; ++k) {
        // Определяем владельца ведущей строки k
        int owner = owner_of_row(k, rowCounts, rowDispls);

        // Владелец копирует ведущую строку в буфер
        if (rank == owner) {
            int localIndex = k - rowDispls[rank];
            double* rowptr = &localA[(size_t)localIndex * N];
            for (int j = 0; j < N; ++j)
                pivotRow[j] = rowptr[j];
            pivotRow[N] = localb[localIndex];
        }

        // Рассылка ведущей строки всем процессам
        MPI_Bcast(pivotRow.data(), N + 1, MPI_DOUBLE,
                  owner, MPI_COMM_WORLD);

        double pivot = pivotRow[k];

        // Защита от вырожденного случая
        if (std::fabs(pivot) < 1e-15) continue;

        // Обновление локальных строк
        for (int li = 0; li < localRows; ++li) {
            int gi = rowDispls[rank] + li; // глобальный индекс строки
            if (gi <= k) continue;

            double* rowptr = &localA[(size_t)li * N];
            double factor = rowptr[k] / pivot;

            // Обновление элементов строки
            for (int j = k; j < N; ++j) {
                rowptr[j] -= factor * pivotRow[j];
            }

            // Обновление правой части
            localb[li] -= factor * pivotRow[N];

            // Обнуляем элемент под диагональю
            rowptr[k] = 0.0;
        }
    }

    // ------------------------------------------------------------
    // 4) Сбор матрицы и вектора b на rank 0
    // ------------------------------------------------------------
    if (rank == 0) {
        A.assign((size_t)N * N, 0.0);
        b.assign(N, 0.0);
    }

    MPI_Gatherv(localA.data(), localRows * N, MPI_DOUBLE,
                rank == 0 ? A.data() : nullptr,
                sendCountsA.data(), sendDisplsA.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(localb.data(), localRows, MPI_DOUBLE,
                rank == 0 ? b.data() : nullptr,
                rowCounts.data(), rowDispls.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    // ------------------------------------------------------------
    // 5) Обратный ход метода Гаусса (только rank 0)
    // ------------------------------------------------------------
    if (rank == 0) {
        std::vector<double> x(N, 0.0);

        for (int i = N - 1; i >= 0; --i) {
            double diag = A[(size_t)i * N + i];
            double rhs  = b[i];

            for (int j = i + 1; j < N; ++j) {
                rhs -= A[(size_t)i * N + j] * x[j];
            }
            x[i] = rhs / diag;
        }

        // Конец замера времени
        double t1 = MPI_Wtime();

        // Вывод результатов
        std::cout << "N = " << N << "\n";
        std::cout << "x[0] = " << x[0] << "\n";
        std::cout << "x[N-1] = " << x[N - 1] << "\n";
        std::cout << "Execution time: "
                  << (t1 - t0) << " seconds\n";
    }

    // Завершение работы MPI
    MPI_Finalize();
    return 0;
}

