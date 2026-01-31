#include <cuda_runtime.h>
#include <iostream>

// CUDA-ядро: каждый поток обрабатывает один элемент
// Доступ коалесцированный — соседние потоки → соседние элементы
__global__ void coalesced(float* a, float* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    b[i] = a[i] * 2.0f;
}

int main() {
    int N = 1 << 20;                 // Размер массива
    size_t size = N * sizeof(float);

    // Выделение памяти на CPU
    float* h_a = (float*)malloc(size);
    float* h_b = (float*)malloc(size);

    // Инициализация массива
    for (int i = 0; i < N; i++)
        h_a[i] = 1.0f;

    // Выделение памяти на GPU
    float *d_a, *d_b;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);

    // Копирование данных CPU → GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Настройка сетки и блоков
    dim3 block(256);
    dim3 grid(N / block.x);

    // Таймер CUDA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    coalesced<<<grid, block>>>(d_a, d_b);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);

    std::cout << "CUDA time: " << ms << " ms\n";
    return 0;
}
