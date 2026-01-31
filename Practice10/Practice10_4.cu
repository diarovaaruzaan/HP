#include <cuda_runtime.h>
#include <iostream>

// Ядро GPU
__global__ void kernel(float* a) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    a[i] *= 2.0f;
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    // Pinned memory на CPU (быстрее для асинхронных копирований)
    float* h;
    cudaMallocHost(&h, size);

    // Память на GPU
    float* d;
    cudaMalloc(&d, size);

    // CUDA stream для асинхронных операций
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Асинхронная передача CPU → GPU
    cudaMemcpyAsync(d, h, size, cudaMemcpyHostToDevice, stream);

    // Запуск ядра в том же stream
    kernel<<<N / 256, 256, 0, stream>>>(d);

    // Асинхронная передача GPU → CPU
    cudaMemcpyAsync(h, d, size, cudaMemcpyDeviceToHost, stream);

    // Ожидание завершения всех операций
    cudaStreamSynchronize(stream);

    std::cout << "Hybrid execution finished\n";
    return 0;
}
