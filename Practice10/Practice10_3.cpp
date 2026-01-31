#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    // Инициализация MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    // Номер процесса
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // Общее количество процессов
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Локальное значение для каждого процесса
    double local = rank + 1;
    double global = 0.0;

    // Измерение времени
    double t1 = MPI_Wtime();

    // Суммирование значений со всех процессов
    MPI_Reduce(&local, &global, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    double t2 = MPI_Wtime();

    // Вывод только на процессе 0
    if (rank == 0)
        std::cout << "Processes: " << size
                  << " | Sum = " << global
                  << " | Time = " << t2 - t1 << std::endl;

    // Завершение MPI
    MPI_Finalize();
    return 0;
}
