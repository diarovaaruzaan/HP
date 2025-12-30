// Assignment2.cpp
#include <iostream>      // Для вывода информации на экран
#include <vector>        // Для работы с динамическими массивами (vector)
#include <algorithm>     // Для swap и других функций
#include <cstdlib>       // Для rand() и srand()
#include <ctime>         // Для времени, чтобы менять seed для rand()
#include <chrono>        // Для измерения времени
#include <omp.h>         // Для работы с OpenMP (параллельные циклы)

// Функция для последовательного поиска min и max
void findMinMaxSequential(const std::vector<int>& arr, int& minVal, int& maxVal) {
    minVal = arr[0];          // Считаем первый элемент минимальным
    maxVal = arr[0];          // Считаем первый элемент максимальным
    for (int i = 1; i < arr.size(); i++) { // Идем по массиву
        if (arr[i] < minVal) minVal = arr[i]; // Если нашли меньше — обновляем min
        if (arr[i] > maxVal) maxVal = arr[i]; // Если нашли больше — обновляем max
    }
}
// Функция для параллельного поиска min и max с OpenMP
void findMinMaxParallel(const std::vector<int>& arr, int& minVal, int& maxVal) {
    minVal = arr[0];
    maxVal = arr[0];

    #pragma omp parallel       // Начало параллельного блока
    {
        int localMin = arr[0]; // У каждого потока своя копия min
        int localMax = arr[0]; // У каждого потока своя копия max

        #pragma omp for nowait // Цикл делится между потоками
        for (int i = 1; i < arr.size(); i++) {
            if (arr[i] < localMin) localMin = arr[i];
            if (arr[i] > localMax) localMax = arr[i];
        }

        #pragma omp critical   // Критическая секция: один поток за раз обновляет глобальные значения
        {
            if (localMin < minVal) minVal = localMin;
            if (localMax > maxVal) maxVal = localMax;
        }
    }
}


// Последовательная сортировка выбором
void selectionSort(std::vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;                 // Изначально минимальный — текущий элемент
        for (int j = i + 1; j < n; j++) // Ищем меньшее значение в оставшейся части массива
            if (arr[j] < arr[minIdx])
                minIdx = j;             // Запоминаем индекс минимального
        std::swap(arr[i], arr[minIdx]); // Меняем текущий элемент с минимальным
    }
}
// Параллельная сортировка выбором с OpenMP
void selectionSortParallel(std::vector<int>& arr) {
    int n = arr.size();
    #pragma omp parallel for schedule(static) // Делим внешний цикл на потоки
    for (int i = 0; i < n - 1; i++) {
        int minIdx = i;
        for (int j = i + 1; j < n; j++)
            if (arr[j] < arr[minIdx])
                minIdx = j;
        #pragma omp critical
        std::swap(arr[i], arr[minIdx]); // Меняем местами в критической секции
    }
}

int main() {
    srand(time(0));                  // Чтобы числа были разные при каждом запуске
    int size = 10000;                // Размер массива
    std::vector<int> arr(size);      // Создаем массив
    for (int i = 0; i < size; i++) arr[i] = rand() % 100000; // Заполняем случайными числами

    int minVal, maxVal;

    // Последовательный поиск min/max
    auto start = std::chrono::high_resolution_clock::now();
    findMinMaxSequential(arr, minVal, maxVal);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Sequential min: " << minVal << ", max: " << maxVal
              << ", time: " << std::chrono::duration<double, std::milli>(end-start).count() << " ms\n";

    // Параллельный поиск min/max
    start = std::chrono::high_resolution_clock::now();
    findMinMaxParallel(arr, minVal, maxVal);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Parallel min: " << minVal << ", max: " << maxVal
              << ", time: " << std::chrono::duration<double, std::milli>(end-start).count() << " ms\n";

    // Сортировка выбором последовательная
    std::vector<int> arrSort = arr;
    start = std::chrono::high_resolution_clock::now();
    selectionSort(arrSort);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Selection sort sequential time: "
              << std::chrono::duration<double, std::milli>(end-start).count() << " ms\n";

    // Сортировка выбором параллельная
    arrSort = arr;
    start = std::chrono::high_resolution_clock::now();
    selectionSortParallel(arrSort);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "Selection sort parallel time: "
              << std::chrono::duration<double, std::milli>(end-start).count() << " ms\n";

    return 0;
}
