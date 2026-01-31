#include <iostream>     // ввод-вывод
#include <vector>       // динамический массив
#include <omp.h>        // OpenMP

int main() {
    // Размер массива
    const int N = 50000000;

    // Инициализация массива значением 1.0
    std::vector<double> a(N, 1.0);

    double sum = 0.0;     // сумма элементов
    double mean = 0.0;    // среднее
    double var = 0.0;     // дисперсия

    // Начало измерения времени
    double t1 = omp_get_wtime();

    // Параллельное вычисление суммы
    // reduction нужен, чтобы корректно суммировать из всех потоков
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++)
        sum += a[i];

    // Среднее значение
    mean = sum / N;

    // Параллельное вычисление дисперсии
    #pragma omp parallel for reduction(+:var)
    for (int i = 0; i < N; i++)
        var += (a[i] - mean) * (a[i] - mean);

    var /= N;

    // Конец измерения времени
    double t2 = omp_get_wtime();

    std::cout << "Time: " << t2 - t1 << " seconds\n";
    return 0;
}

