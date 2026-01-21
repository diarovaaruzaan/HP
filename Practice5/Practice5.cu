#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <functional>   // используется для std::function в замерах времени

// ============================================================
// Макрос для проверки ошибок CUDA
// Если CUDA-функция вернула ошибку — программа завершится
// ============================================================
#define CUDA_CHECK(call) do {                                  \
    cudaError_t err = (call);                                  \
    if (err != cudaSuccess) {                                  \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << std::endl;                                \
        std::exit(1);                                          \
    }                                                          \
} while(0)

// Проверка ошибок после запуска kernel
#define CUDA_KERNEL_CHECK() do {            \
    CUDA_CHECK(cudaGetLastError());         \
    CUDA_CHECK(cudaDeviceSynchronize());    \
} while(0)

// ============================================================
// ПАРАЛЛЕЛЬНЫЙ СТЕК (LIFO)
// Последний добавленный элемент извлекается первым
// Реализован с использованием атомарных операций
// ============================================================
struct Stack {
    int* data;        // массив данных в глобальной памяти GPU
    int top;          // количество элементов в стеке
    int capacity;     // максимальный размер стека

    // Инициализация стека (вызывается на GPU)
    __device__ void init(int* buffer, int size) {
        data = buffer;
        top = 0;
        capacity = size;
    }

    // Добавление элемента в стек
    __device__ bool push(int value) {
        // атомарно увеличиваем top и получаем позицию записи
        int pos = atomicAdd(&top, 1);

        // если стек не переполнен — записываем значение
        if (pos < capacity) {
            data[pos] = value;
            return true;
        }

        // если переполнение — откатываем top назад
        atomicSub(&top, 1);
        return false;
    }

    // Извлечение элемента из стека
    __device__ bool pop(int* value) {
        // атомарно уменьшаем top и получаем позицию чтения
        int pos = atomicSub(&top, 1) - 1;

        if (pos >= 0) {
            *value = data[pos];
            return true;
        }

        // если стек был пуст — возвращаем top обратно
        atomicAdd(&top, 1);
        return false;
    }
};

// ============================================================
// ПАРАЛЛЕЛЬНАЯ ОЧЕРЕДЬ (FIFO)
// Первый добавленный элемент извлекается первым
// ============================================================
struct Queue {
    int* data;        // массив данных
    int head;         // индекс чтения
    int tail;         // индекс записи
    int capacity;     // максимальный размер очереди

    // Инициализация очереди
    __device__ void init(int* buffer, int size) {
        data = buffer;
        head = 0;
        tail = 0;
        capacity = size;
    }

    // Добавление элемента в очередь
    __device__ bool enqueue(int value) {
        int pos = atomicAdd(&tail, 1);

        if (pos < capacity) {
            data[pos] = value;
            return true;
        }

        // если очередь переполнена — откат tail
        atomicSub(&tail, 1);
        return false;
    }

    // Извлечение элемента из очереди
    __device__ bool dequeue(int* value) {
        int pos = atomicAdd(&head, 1);

        if (pos < tail) {
            *value = data[pos];
            return true;
        }

        // если очередь пуста — откат head
        atomicSub(&head, 1);
        return false;
    }
};

// ============================================================
// Ядра инициализации структур данных
// Выполняются одним потоком
// ============================================================
__global__ void initStackKernel(Stack* s, int* buffer, int cap) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        s->init(buffer, cap);
}

__global__ void initQueueKernel(Queue* q, int* buffer, int cap) {
    if (threadIdx.x == 0 && blockIdx.x == 0)
        q->init(buffer, cap);
}

// ============================================================
// Ядра для проверки корректности работы
// ============================================================
__global__ void stackPushKernel(Stack* s, int nOps, int* ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nOps)
        ok[tid] = s->push(tid) ? 1 : 0;
}

__global__ void stackPopKernel(Stack* s, int nOps, int* out, int* ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nOps) {
        int v = -1;
        bool success = s->pop(&v);
        ok[tid] = success ? 1 : 0;
        out[tid] = v;
    }
}

__global__ void queueEnqKernel(Queue* q, int nOps, int* ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nOps)
        ok[tid] = q->enqueue(tid) ? 1 : 0;
}

__global__ void queueDeqKernel(Queue* q, int nOps, int* out, int* ok) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < nOps) {
        int v = -1;
        bool success = q->dequeue(&v);
        ok[tid] = success ? 1 : 0;
        out[tid] = v;
    }
}

// ============================================================
// Вспомогательные функции на CPU
// ============================================================

// Подсчёт количества успешных операций
long long sum01(const std::vector<int>& a) {
    long long s = 0;
    for (int x : a) s += x;
    return s;
}

// Проверка, что все значения 0..n-1 встречаются ровно один раз
bool checkPermutation(const std::vector<int>& out, int n) {
    std::vector<int> cnt(n, 0);
    for (int v : out) {
        if (v < 0 || v >= n) return false;
        cnt[v]++;
    }
    for (int i = 0; i < n; i++)
        if (cnt[i] != 1) return false;
    return true;
}

// Замер времени выполнения kernel с помощью cudaEvent
float timeKernel(std::function<void()> launch, int warmup = 3, int iters = 10) {
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Прогрев GPU
    for (int i = 0; i < warmup; i++) {
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++)
        launch();
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms / iters;
}
