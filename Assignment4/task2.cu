#include <cuda_runtime.h>   // Основные функции CUDA Runtime API
#include <iostream>         // cout, cerr
#include <vector>           // std::vector
#include <random>           // генератор случайных чисел
#include <chrono>           // измерение времени на CPU
#include <iomanip>          // форматированный вывод
#include <algorithm>        // std::max
#include <cmath>            // std::abs

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

/*
 * CUDA-ядро: префиксная сумма (inclusive scan) в shared memory.
 * Каждый поток обрабатывает 2 элемента массива.
 * Вычисления выполняются в double для высокой точности.
 */
__global__ void scan_block_inclusive_2x_double(
    const float* __restrict__ in,     // входной массив (float)
    double* __restrict__ out,          // выходной массив (double)
    double* __restrict__ blockSums,    // суммы блоков
    int n                              // размер массива
)
{
    // Выделяем shared memory (2 * blockDim элементов)
    extern __shared__ double s[];

    // Локальный индекс потока внутри блока
    int tid = threadIdx.x;

    // Начальный индекс текущего блока в глобальном массиве
    int base = 2 * blockIdx.x * blockDim.x;

    // Глобальные индексы двух элементов, которые обрабатывает поток
    int i0 = base + tid;
    int i1 = base + tid + blockDim.x;

    // Загружаем данные из глобальной памяти
    // Если индекс выходит за границу — берём 0
    double x0 = (i0 < n) ? (double)in[i0] : 0.0;
    double x1 = (i1 < n) ? (double)in[i1] : 0.0;

    // Записываем данные в shared memory
    s[tid] = x0;
    s[tid + blockDim.x] = x1;

    // Ждём, пока все потоки загрузят данные
    __syncthreads();

    // Общее количество элементов в блоке
    int m = 2 * blockDim.x;

    /* ===== UPSWEEP (reduce-фаза) =====
       Строим дерево сумм в shared memory
    */
    for (int offset = 1; offset < m; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < m)
            s[idx] += s[idx - offset];
        __syncthreads();
    }

    // Последний элемент содержит сумму всего блока
    if (tid == 0) {
        blockSums[blockIdx.x] = s[m - 1]; // сохраняем сумму блока
        s[m - 1] = 0.0;                   // подготавливаем для downsweep
    }
    __syncthreads();

    /* ===== DOWNSWEEP (exclusive scan) ===== */
    for (int offset = m >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;
        if (idx < m) {
            double t = s[idx - offset];
            s[idx - offset] = s[idx];
            s[idx] += t;
        }
        __syncthreads();
    }

    // Преобразуем exclusive scan → inclusive scan
    if (i0 < n) out[i0] = s[tid] + x0;
    if (i1 < n) out[i1] = s[tid + blockDim.x] + x1;
}

/*
 * CUDA-ядро: добавление смещений блоков
 * Каждому элементу блока прибавляется сумма всех предыдущих блоков
 */
__global__ void add_block_offsets_2x_double(
    double* __restrict__ out,
    const double* __restrict__ blockOffsets,
    int n
)
{
    int tid = threadIdx.x;
    int base = 2 * blockIdx.x * blockDim.x;

    int i0 = base + tid;
    int i1 = base + tid + blockDim.x;

    double offset = blockOffsets[blockIdx.x];

    if (i0 < n) out[i0] += offset;
    if (i1 < n) out[i1] += offset;
}

/*
 * Последовательная CPU-версия префиксной суммы (inclusive scan)
 * Используется как эталон для проверки корректности
 */
static void cpu_scan_inclusive_double(
    const std::vector<float>& in,
    std::vector<double>& out
)
{
    out.resize(in.size());
    double acc = 0.0;

    for (size_t i = 0; i < in.size(); ++i) {
        acc += (double)in[i];
        out[i] = acc;
    }
}

int main()
{
    // Размер массива
    const int N = 1'000'000;

    // Количество потоков в блоке (степень двойки!)
    const int blockSize = 256;

    // Количество элементов, обрабатываемых одним блоком
    const int elemsPerBlock = 2 * blockSize;

    // Получаем информацию о GPU
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "GPU: " << prop.name << "\n";

    // Генерация входных данных
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> h_in(N);
    for (int i = 0; i < N; ++i)
        h_in[i] = dist(rng);

    /* ===== CPU ===== */
    std::vector<double> h_cpu;
    auto c0 = std::chrono::high_resolution_clock::now();
    cpu_scan_inclusive_double(h_in, h_cpu);
    auto c1 = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(c1 - c0).count();

    /* ===== GPU ===== */
    float*  d_in = nullptr;
    double* d_out = nullptr;
    double* d_blockSums = nullptr;
    double* d_blockOffsets = nullptr;

    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(),
                          N * sizeof(float),
                          cudaMemcpyHostToDevice));

    int numBlocks = (N + elemsPerBlock - 1) / elemsPerBlock;

    CUDA_CHECK(cudaMalloc(&d_blockSums,   numBlocks * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_blockOffsets,numBlocks * sizeof(double)));

    // GPU таймер
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventRecord(e0));

    // 1) Prefix sum внутри блоков (shared memory)
    scan_block_inclusive_2x_double<<<
        numBlocks,
        blockSize,
        elemsPerBlock * sizeof(double)
    >>>(d_in, d_out, d_blockSums, N);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2) Смещения блоков считаем на CPU
    std::vector<double> h_sums(numBlocks), h_offsets(numBlocks);
    CUDA_CHECK(cudaMemcpy(h_sums.data(), d_blockSums,
                          numBlocks * sizeof(double),
                          cudaMemcpyDeviceToHost));

    double acc = 0.0;
    for (int b = 0; b < numBlocks; ++b) {
        h_offsets[b] = acc;
        acc += h_sums[b];
    }

    CUDA_CHECK(cudaMemcpy(d_blockOffsets, h_offsets.data(),
                          numBlocks * sizeof(double),
                          cudaMemcpyHostToDevice));

    // 3) Добавляем смещения блоков
    add_block_offsets_2x_double<<<numBlocks, blockSize>>>(
        d_out, d_blockOffsets, N
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));

    float gpu_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, e0, e1));

    // Копируем результат
    std::vector<double> h_gpu(N);
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out,
                          N * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // Проверка корректности
    double max_diff = 0.0;
    for (int i = 0; i < N; ++i)
        max_diff = std::max(max_diff, std::abs(h_cpu[i] - h_gpu[i]));

    // Освобождение памяти
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_blockSums));
    CUDA_CHECK(cudaFree(d_blockOffsets));
    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));

    // Вывод результатов
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Task 2 — Prefix sum (scan) using shared memory (double)\n";
    std::cout << "CPU time ms: " << cpu_ms << "\n";
    std::cout << "GPU time ms: " << gpu_ms << "\n";
    std::cout << "Max abs diff: " << max_diff << "\n";

    return 0;
}
