#include <cuda_runtime.h>   // Основной заголовок CUDA Runtime API
#include <iostream>         // Ввод/вывод (cout, cerr)
#include <vector>           // Контейнер std::vector
#include <random>           // Генерация случайных чисел
#include <numeric>          // std::accumulate
#include <chrono>           // Замеры времени на CPU
#include <iomanip>          // Форматированный вывод

// Макрос для удобной проверки ошибок CUDA-вызовов
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

/*
 * CUDA-ядро для вычисления суммы элементов массива.
 * Каждый поток обрабатывает один элемент массива и
 * добавляет его значение в общую переменную outSum
 * с использованием атомарной операции atomicAdd.
 *
 * Используется глобальная память GPU.
 */
__global__ void sum_atomic_global(const float* __restrict__ x,
                                  float* __restrict__ outSum,
                                  int n)
{
    // Глобальный индекс потока
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверяем, что индекс не выходит за границы массива
    if (i < n) {
        // Атомарно прибавляем значение элемента массива
        // к общей сумме в глобальной памяти
        atomicAdd(outSum, x[i]);
    }
}

int main()
{
    // Размер массива
    const int N = 100000;

    // Количество потоков в одном блоке
    const int blockSize = 256;

    /* =======================
       Проверка наличия GPU
       ======================= */
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }

    // Получаем и выводим информацию о видеокарте
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    /* =======================
       Подготовка данных
       ======================= */
    std::mt19937 rng(42);  // Генератор случайных чисел
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Хост-массив (CPU)
    std::vector<float> h_x(N);
    for (int i = 0; i < N; ++i)
        h_x[i] = dist(rng);

    /* =======================
       Последовательная версия (CPU)
       ======================= */
    auto t0 = std::chrono::high_resolution_clock::now();

    // Суммирование элементов массива на CPU
    // Используется тип double для большей точности
    double cpu_sum = std::accumulate(h_x.begin(), h_x.end(), 0.0);

    auto t1 = std::chrono::high_resolution_clock::now();

    // Время выполнения CPU-версии
    double cpu_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    /* =======================
       CUDA-версия (GPU)
       ======================= */
    float *d_x = nullptr;    // Указатель на массив на GPU
    float *d_sum = nullptr;  // Указатель на сумму на GPU

    // Выделяем память на GPU под массив
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(float)));

    // Выделяем память под переменную суммы
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));

    // Копируем данные массива с CPU на GPU
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Обнуляем переменную суммы на GPU
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));

    // Количество блоков в сетке
    int grid = (N + blockSize - 1) / blockSize;

    // CUDA-события для измерения времени работы GPU
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // Запуск таймера
    CUDA_CHECK(cudaEventRecord(e0));

    // Запуск CUDA-ядра
    sum_atomic_global<<<grid, blockSize>>>(d_x, d_sum, N);

    // Проверка ошибок запуска ядра
    CUDA_CHECK(cudaGetLastError());

    // Ожидание завершения всех потоков GPU
    CUDA_CHECK(cudaDeviceSynchronize());

    // Остановка таймера
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    // Получаем время выполнения GPU-версии
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    // Копируем результат суммы с GPU на CPU
    float h_sum_f = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum_f, d_sum,
                          sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Освобождаем память GPU
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    /* =======================
       Сравнение результатов
       ======================= */
    double gpu_sum = static_cast<double>(h_sum_f);
    double diff = std::abs(cpu_sum - gpu_sum);

    // Вывод результатов
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Task 1 — Sum using CUDA global memory (atomicAdd)\n";
    std::cout << "CPU sum: " << cpu_sum
              << ", time ms: " << cpu_ms << "\n";
    std::cout << "GPU sum: " << gpu_sum
              << ", time ms: " << gpu_ms << "\n";
    std::cout << "Abs diff: " << diff << "\n";

    return 0;
}

