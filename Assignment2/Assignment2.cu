#include <iostream>               // Для вывода на экран
#include <vector>                 // Для std::vector
#include <cstdlib>                // Для rand()
#include <ctime>                  // Для srand()
#include <cuda_runtime.h>         // Для работы с CUDA
#include <device_launch_parameters.h> // Для blockIdx, threadIdx и др.

// -----------------------------
// CUDA Kernel: слияние двух подмассивов
// -----------------------------
__global__ void mergeKernel(int* d_arr, int* d_tmp, int n, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Индекс потока
    int start = idx * width * 2;                     // Начало подмассива
    if (start >= n) return;                          // Если вышли за массив — выходим

    int mid = min(start + width, n);                // Середина подмассива
    int end = min(start + width * 2, n);            // Конец подмассива

    int i = start, j = mid, k = start;
    while (i < mid && j < end) {                    // Сливаем два подмассива
        if (d_arr[i] <= d_arr[j]) d_tmp[k++] = d_arr[i++];
        else d_tmp[k++] = d_arr[j++];
    }
    while (i < mid) d_tmp[k++] = d_arr[i++];       // Копируем остатки первого подмассива
    while (j < end) d_tmp[k++] = d_arr[j++];       // Копируем остатки второго подмассива
}

// -----------------------------
// Функция сортировки слиянием на GPU
// -----------------------------
void mergeSortCUDA(std::vector<int>& arr) {
    int n = arr.size();
    int* d_arr;
    int* d_tmp;

    // Выделяем память на GPU
    cudaMalloc(&d_arr, n * sizeof(int));
    cudaMalloc(&d_tmp, n * sizeof(int));
    // Копируем массив с CPU на GPU
    cudaMemcpy(d_arr, arr.data(), n * sizeof(int), cudaMemcpyHostToDevice);

    int width = 1;                   // Начальный размер подмассива
    int threadsPerBlock = 512;       // Потоки на блок

    while (width < n) {
        int blocks = (n / (2 * width) + threadsPerBlock - 1) / threadsPerBlock;
        mergeKernel<<<blocks, threadsPerBlock>>>(d_arr, d_tmp, n, width);
        cudaDeviceSynchronize();     // Ждём завершения всех потоков
        std::swap(d_arr, d_tmp);     // Меняем массивы местами
        width *= 2;                  // Увеличиваем размер подмассива в 2 раза
    }

    // Копируем отсортированный массив обратно на CPU
    cudaMemcpy(arr.data(), d_arr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождаем память
    cudaFree(d_arr);
    cudaFree(d_tmp);
}

// -----------------------------
// Главная функция
// -----------------------------
int main() {
    srand(time(0));

    std::vector<int> sizes = {10000, 100000}; // Размеры массивов

    for (int size : sizes) {
        std::vector<int> arr(size);

        // Заполняем массив случайными числами
        for (int i = 0; i < size; i++)
            arr[i] = rand() % 100000;

        // -----------------------------
        // Используем CUDA Events для точного замера времени на GPU
        // -----------------------------
        cudaEvent_t startEvent, stopEvent;
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);

        cudaEventRecord(startEvent, 0);  // Засекаем начало
        mergeSortCUDA(arr);               // Сортировка на GPU
        cudaEventRecord(stopEvent, 0);   // Засекаем конец
        cudaEventSynchronize(stopEvent); // Ждем завершения всех потоков

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent); // Время в мс

        // Выводим результат
        std::cout << "CUDA sort for array of size " << size
                  << " tok " << milliseconds << " ms\n";

        // Удаляем события
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    return 0;
}
