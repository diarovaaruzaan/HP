#include <iostream>          // библиотека для вывода информации (cout)
#include <cuda_runtime.h>    // библиотека CUDA для работы с GPU
#include <algorithm>         // стандартные алгоритмы (swap, min)
#include <chrono>            // библиотека для измерения времени
#include <vector>            // библиотека для динамических массивов vector
#include <cstdlib>           // библиотека для rand()

using namespace std;         // чтобы не писать std:: перед каждым элементом

// ==========================================================
// ---------------- GPU MERGE SORT (CUDA) -------------------
// ==========================================================

// CUDA kernel — выполняется на GPU
// Каждый поток сливает два подмассива
__global__ void mergeKernel(int* d_arr, int* d_temp, int size, int width) {

    // Глобальный индекс потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Начало участка массива для данного потока
    int start = idx * (2 * width);

    // Если поток выходит за границы массива — ничего не делаем
    if (start >= size) return;

    // Определяем середину и конец подмассива
    int mid = min(start + width, size);
    int end = min(start + 2 * width, size);

    // Индексы для слияния
    int i = start;   // левая часть
    int j = mid;     // правая часть
    int k = start;   // временный массив

    // Сливаем элементы из двух частей
    while (i < mid && j < end) {
        if (d_arr[i] <= d_arr[j])
            d_temp[k++] = d_arr[i++];
        else
            d_temp[k++] = d_arr[j++];
    }

    // Если остались элементы в левой части
    while (i < mid)
        d_temp[k++] = d_arr[i++];

    // Если остались элементы в правой части
    while (j < end)
        d_temp[k++] = d_arr[j++];
}

// Функция сортировки слиянием на GPU
void mergeSortCUDA(int* arr, int size) {

    int* d_arr;   // массив на GPU
    int* d_temp;  // временный массив на GPU

    // Выделяем память на GPU
    cudaMalloc(&d_arr, size * sizeof(int));
    cudaMalloc(&d_temp, size * sizeof(int));

    // Копируем данные с CPU на GPU
    cudaMemcpy(d_arr, arr, size * sizeof(int), cudaMemcpyHostToDevice);

    // Начальный размер подмассивов
    int width = 1;

    // Пока размер подмассивов меньше размера массива
    while (width < size) {

        // Количество потоков
        int threads = (size + 2 * width - 1) / (2 * width);

        // Запуск CUDA kernel
        mergeKernel<<<(threads + 255) / 256, 256>>>(d_arr, d_temp, size, width);

        // Ждём завершения всех потоков
        cudaDeviceSynchronize();

        // Меняем массивы местами
        swap(d_arr, d_temp);

        // Увеличиваем размер подмассивов
        width *= 2;
    }

    // Копируем результат обратно на CPU
    cudaMemcpy(arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Освобождаем память GPU
    cudaFree(d_arr);
    cudaFree(d_temp);
}

// ==========================================================
// ---------------- CPU MERGE SORT --------------------------
// ==========================================================

// Рекурсивная сортировка слиянием на CPU
void mergeCPU(vector<int>& arr, int l, int r) {

    // Если один элемент — сортировка не нужна
    if (l >= r) return;

    // Находим середину массива
    int m = (l + r) / 2;

    // Сортируем левую и правую часть
    mergeCPU(arr, l, m);
    mergeCPU(arr, m + 1, r);

    // Временный массив для слияния
    vector<int> temp(r - l + 1);

    int i = l, j = m + 1, k = 0;

    // Слияние двух отсортированных частей
    while (i <= m && j <= r) {
        if (arr[i] <= arr[j]) temp[k++] = arr[i++];
        else temp[k++] = arr[j++];
    }

    // Остатки
    while (i <= m) temp[k++] = arr[i++];
    while (j <= r) temp[k++] = arr[j++];

    // Копируем обратно
    for (int x = 0; x < temp.size(); x++)
        arr[l + x] = temp[x];
}

// ==========================================================
// ---------------- CPU QUICK SORT --------------------------
// ==========================================================

void quickSortCPU(vector<int>& arr, int left, int right) {

    if (left >= right) return;

    int pivot = arr[(left + right) / 2];
    int l = left, r = right;

    while (l <= r) {
        while (arr[l] < pivot) l++;
        while (arr[r] > pivot) r--;

        if (l <= r) {
            swap(arr[l], arr[r]);
            l++;
            r--;
        }
    }

    quickSortCPU(arr, left, r);
    quickSortCPU(arr, l, right);
}

// ==========================================================
// ---------------- CPU HEAP SORT ---------------------------
// ==========================================================

void heapify(vector<int>& arr, int n, int i) {

    int largest = i;
    int l = 2 * i + 1;
    int r = 2 * i + 2;

    if (l < n && arr[l] > arr[largest]) largest = l;
    if (r < n && arr[r] > arr[largest]) largest = r;

    if (largest != i) {
        swap(arr[i], arr[largest]);
        heapify(arr, n, largest);
    }
}

void heapSortCPU(vector<int>& arr) {

    int n = arr.size();

    // Строим кучу
    for (int i = n / 2 - 1; i >= 0; i--)
        heapify(arr, n, i);

    // Извлекаем элементы
    for (int i = n - 1; i >= 0; i--) {
        swap(arr[0], arr[i]);
        heapify(arr, i, 0);
    }
}

// ==========================================================
// ------------------------ MAIN ----------------------------
// ==========================================================

int main() {

    // Размеры массивов для тестирования
    int sizes[] = {10000, 100000, 1000000};

    for (int s = 0; s < 3; s++) {

        int size = sizes[s];
        cout << "\nArray size: " << size << endl;

        // Исходный массив
        vector<int> arr(size);
        for (int i = 0; i < size; i++)
            arr[i] = rand() % 100000;

        // Копии массива
        vector<int> a1 = arr, a2 = arr, a3 = arr;

        // CPU Merge Sort
        auto start = chrono::high_resolution_clock::now();
        mergeCPU(a1, 0, size - 1);
        auto end = chrono::high_resolution_clock::now();
        cout << "CPU Merge Sort: "
             << chrono::duration_cast<chrono::milliseconds>(end - start).count()
             << " ms\n";

        // CPU Quick Sort
        start = chrono::high_resolution_clock::now();
        quickSortCPU(a2, 0, size - 1);
        end = chrono::high_resolution_clock::now();
        cout << "CPU Quick Sort: "
             << chrono::duration_cast<chrono::milliseconds>(end - start).count()
             << " ms\n";

        // CPU Heap Sort
        start = chrono::high_resolution_clock::now();
        heapSortCPU(a3);
        end = chrono::high_resolution_clock::now();
        cout << "CPU Heap Sort: "
             << chrono::duration_cast<chrono::milliseconds>(end - start).count()
             << " ms\n";

        // GPU Merge Sort
        int* arrGPU = new int[size];
        for (int i = 0; i < size; i++) arrGPU[i] = arr[i];

        start = chrono::high_resolution_clock::now();
        mergeSortCUDA(arrGPU, size);
        end = chrono::high_resolution_clock::now();
        cout << "GPU Merge Sort: "
             << chrono::duration_cast<chrono::milliseconds>(end - start).count()
             << " ms\n";

        delete[] arrGPU;
    }

    return 0;
}
