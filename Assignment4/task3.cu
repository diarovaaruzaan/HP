#include <cuda_runtime.h>   // CUDA Runtime API
#include <iostream>         // cout/cerr
#include <vector>           // std::vector
#include <random>           // генерация случайных чисел
#include <chrono>           // время CPU
#include <iomanip>          // setprecision
#include <algorithm>        // std::max
#include <cmath>            // std::abs

// Макрос проверки ошибок CUDA: если ошибка — печатаем и выходим
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

/*
  GPU kernel: "обработка массива" (пример из типовых заданий)
  y[i] = a * x[i] + b

  Это хороший пример, потому что:
  - одинаковая логика на CPU и GPU
  - легко сравнивать результаты
  - операция параллелится по элементам
*/
__global__ void affine_kernel(const float* __restrict__ x,
                              float* __restrict__ y,
                              int n,
                              float a,
                              float b)
{
    // Глобальный индекс элемента, который обрабатывает поток
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Проверка границ (последний блок может быть неполным)
    if (i < n) {
        // Основная формула обработки
        y[i] = a * x[i] + b;
    }
}

/*
  CPU обработка: то же самое, но последовательно.
  Используем для сравнения времени и для проверки корректности.
*/
static void cpu_affine(const float* x, float* y, int n, float a, float b)
{
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + b;
    }
}

int main()
{
    // Размер массива — можно менять.
    // Если поставить совсем маленький, гибрид может не дать выигрыша из-за копирований.
    const int N = 5'000'000;

    // Делим массив пополам: первая половина CPU, вторая GPU
    const int half = N / 2;

    // Параметры формулы y = a*x + b
    const float a = 1.7f;
    const float b = 0.3f;

    // Размер блока CUDA (часто 256 — хороший стандарт)
    const int blockSize = 256;

    /* ========== Проверка GPU ========== */
    int devCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devCount));
    if (devCount == 0) {
        std::cerr << "No CUDA devices found.\n";
        return 1;
    }
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    /* ========== Генерация входных данных ========== */
    std::mt19937 rng(42); // фиксированный seed, чтобы результаты были воспроизводимыми
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Входной массив на CPU
    std::vector<float> x(N);

    // Результаты:
    // - CPU-only
    // - GPU-only
    // - Hybrid
    std::vector<float> y_cpu(N);
    std::vector<float> y_gpu(N);
    std::vector<float> y_hybrid(N);

    // Заполняем входной массив случайными числами
    for (int i = 0; i < N; ++i) {
        x[i] = dist(rng);
    }

    /* ============================================================
       1) CPU-only: вся обработка на CPU
       ============================================================ */
    auto cpu_t0 = std::chrono::high_resolution_clock::now();

    // Обрабатываем весь массив на CPU
    cpu_affine(x.data(), y_cpu.data(), N, a, b);

    auto cpu_t1 = std::chrono::high_resolution_clock::now();

    // Считаем время в миллисекундах
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();

    /* ============================================================
       2) GPU-only: вся обработка на GPU
       (время считаем через cudaEvent, это стандарт для измерений GPU)
       ============================================================ */

    // Выделяем память на GPU под вход и выход
    float* d_x = nullptr;
    float* d_y = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, (size_t)N * sizeof(float)));

    // Копируем входной массив на GPU
    CUDA_CHECK(cudaMemcpy(d_x, x.data(),
                          (size_t)N * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Создаём события для замера времени GPU
    cudaEvent_t g0, g1;
    CUDA_CHECK(cudaEventCreate(&g0));
    CUDA_CHECK(cudaEventCreate(&g1));

    // Запускаем таймер
    CUDA_CHECK(cudaEventRecord(g0));

    // Количество блоков в сетке
    int grid = (N + blockSize - 1) / blockSize;

    // Запускаем kernel на весь массив
    affine_kernel<<<grid, blockSize>>>(d_x, d_y, N, a, b);

    // Проверяем ошибки запуска kernel
    CUDA_CHECK(cudaGetLastError());

    // Ждём завершения kernel (иначе время и данные будут некорректными)
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копируем результат обратно на CPU
    CUDA_CHECK(cudaMemcpy(y_gpu.data(), d_y,
                          (size_t)N * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Останавливаем таймер
    CUDA_CHECK(cudaEventRecord(g1));
    CUDA_CHECK(cudaEventSynchronize(g1));

    // Считаем время GPU
    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, g0, g1));

    /* ============================================================
       3) HYBRID: половина на CPU, половина на GPU
       Здесь измеряем "общее время" на CPU-таймере:
       - CPU вычисления первой половины
       - копирование второй половины на GPU
       - kernel на второй половине
       - копирование результата второй половины назад
       ============================================================ */
    auto h0 = std::chrono::high_resolution_clock::now();

    // 3.1 CPU обрабатывает первую половину [0 .. half-1]
    cpu_affine(x.data(), y_hybrid.data(), half, a, b);

    // 3.2 GPU обрабатывает вторую половину [half .. N-1]
    int n2 = N - half; // количество элементов второй половины

    // Копируем вторую половину входа в d_x (начиная с 0 в GPU-буфере)
    CUDA_CHECK(cudaMemcpy(d_x, x.data() + half,
                          (size_t)n2 * sizeof(float),
                          cudaMemcpyHostToDevice));

    // Запускаем kernel на n2 элементах
    int grid2 = (n2 + blockSize - 1) / blockSize;
    affine_kernel<<<grid2, blockSize>>>(d_x, d_y, n2, a, b);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Копируем результат второй половины назад в y_hybrid[half..]
    CUDA_CHECK(cudaMemcpy(y_hybrid.data() + half, d_y,
                          (size_t)n2 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    auto h1 = std::chrono::high_resolution_clock::now();

    // Полное время гибридного варианта
    double hybrid_ms =
        std::chrono::duration<double, std::milli>(h1 - h0).count();

    /* ============================================================
       Проверка корректности (несколько точек)
       Сравним Hybrid и CPU-only (они должны совпадать)
       ============================================================ */
    double max_abs_diff = 0.0;

    // Проверяем несколько индексов, включая границу половины
    int test_idx[] = {0, 1, 2, half - 1, half, half + 1, N - 2, N - 1};

    for (int idx : test_idx) {
        double d = std::abs((double)y_cpu[idx] - (double)y_hybrid[idx]);
        max_abs_diff = std::max(max_abs_diff, d);
    }

    /* ========== Освобождение ресурсов GPU ========== */
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(g0));
    CUDA_CHECK(cudaEventDestroy(g1));

    /* ========== Вывод результатов ========== */
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Task 3 — Hybrid CPU+GPU array processing\n";
    std::cout << "N = " << N << " (first half CPU, second half GPU)\n";
    std::cout << "CPU-only   time ms: " << cpu_ms << "\n";
    std::cout << "GPU-only   time ms: " << gpu_ms << "\n";
    std::cout << "Hybrid     time ms: " << hybrid_ms << "\n";

    std::cout << std::setprecision(6);
    std::cout << "Max abs diff (CPU vs Hybrid, sample points): " << max_abs_diff << "\n";

    return 0;
}
