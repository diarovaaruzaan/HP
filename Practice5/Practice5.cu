%%writefile main.cu
// ========================== main.cu ========================== // Файл CUDA-программы (компилируется nvcc)
#include <cuda_runtime.h>                 // CUDA runtime API (cudaMalloc, cudaMemcpy, cudaEvent и т.д.)
#include <device_launch_parameters.h>     // Макросы/типы для запуска kernel (blockIdx, threadIdx)

#include <iostream>                       // std::cout, std::cerr для вывода
#include <vector>                         // std::vector для массивов на CPU
#include <functional>                     // std::function для передачи лямбды в таймер

// ============================================================
// Макрос: проверка ошибок CUDA (если ошибка — печать и выход)
// ============================================================
#define CUDA_CHECK(call) do {                                                  \
    cudaError_t err = (call);                                                  /* выполняем CUDA-вызов и сохраняем код ошибки */ \
    if (err != cudaSuccess) {                                                  /* если ошибка не равна cudaSuccess */ \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)                 /* печатаем текст ошибки */ \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl;       /* печатаем файл и строку */ \
        std::exit(1);                                                          /* завершаем программу с кодом 1 */ \
    }                                                                          \
} while (0)                                                                     /* конструкция макроса как единый оператор */

// ============================================================
// CPU: посчитать сумму 0/1 (сколько успешных операций)
// ============================================================
static long long sum01(const std::vector<int>& a) {                             // функция принимает вектор (0/1)
    long long s = 0;                                                           // сумма успешных операций
    for (int x : a) s += x;                                                    // прибавляем каждый элемент
    return s;                                                                  // возвращаем сумму
}

// ============================================================
// CPU: проверить, что out содержит все значения 0..n-1 ровно по 1 разу
// (Порядок не важен — это нормально для параллельности)
// ============================================================
static bool checkPermutation(const std::vector<int>& out, int n) {              // out — результаты, n — ожидаемый диапазон
    std::vector<int> cnt(n, 0);                                                // счётчик вхождений для каждого числа 0..n-1
    for (int v : out) {                                                        // идём по всем извлечённым значениям
        if (v < 0 || v >= n) return false;                                     // если значение вне диапазона — плохо
        cnt[v]++;                                                              // увеличиваем счётчик для v
    }
    for (int i = 0; i < n; i++) {                                              // проверяем каждый счётчик
        if (cnt[i] != 1) return false;                                         // должно быть ровно 1
    }
    return true;                                                               // если все ок — перестановка корректна
}

// ============================================================
// CPU: замер времени kernel через cudaEvent
// launch() должен запускать kernel, синхронизацию делаем внутри
// ============================================================
static float timeKernel(const std::function<void()>& launch,                    // launch — функция, запускающая kernel
                        int warmup = 3,                                         // warmup — прогрев (чтобы убрать холодный старт)
                        int iters  = 10) {                                      // iters — сколько раз измерять
    cudaEvent_t start, stop;                                                   // события CUDA для таймера
    CUDA_CHECK(cudaEventCreate(&start));                                        // создаём событие старта
    CUDA_CHECK(cudaEventCreate(&stop));                                         // создаём событие окончания

    for (int i = 0; i < warmup; i++) {                                         // прогрев GPU
        launch();                                                              // запускаем kernel
        CUDA_CHECK(cudaDeviceSynchronize());                                   // ждём завершения (чтобы прогрев был честный)
    }

    CUDA_CHECK(cudaEventRecord(start));                                        // ставим событие "старт"
    for (int i = 0; i < iters; i++) {                                          // несколько повторов для усреднения
        launch();                                                              // запускаем kernel
    }
    CUDA_CHECK(cudaEventRecord(stop));                                         // ставим событие "стоп"
    CUDA_CHECK(cudaEventSynchronize(stop));                                    // ждём, пока stop реально наступит

    float ms = 0.0f;                                                           // сюда запишем время в миллисекундах
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));                        // считаем разницу между start и stop
    CUDA_CHECK(cudaEventDestroy(start));                                       // удаляем событие start
    CUDA_CHECK(cudaEventDestroy(stop));                                        // удаляем событие stop

    return ms / iters;                                                         // возвращаем среднее время за 1 запуск
}

// ============================================================
// Параллельный стек (LIFO) с "публикацией" данных через ready[]
// ============================================================
struct Stack {                                                                 // структура стека (на GPU)
    int* data;                                                                 // указатель на массив значений (global memory)
    int* ready;                                                                // указатель на массив флагов готовности (global memory)
    int  top;                                                                  // количество элементов (индекс следующей записи)
    int  capacity;                                                             // максимальная ёмкость

    __device__ void init(int* buffer, int* readyBuf, int size) {               // инициализация на GPU
        data = buffer;                                                         // запоминаем адрес данных
        ready = readyBuf;                                                      // запоминаем адрес флагов
        top = 0;                                                               // стек пуст
        capacity = size;                                                       // сохраняем ёмкость
    }

    __device__ bool push(int value) {                                          // операция push на GPU
        int pos = atomicAdd(&top, 1);                                          // атомарно резервируем позицию и увеличиваем top
        if (pos < capacity) {                                                  // если не вышли за ёмкость
            data[pos] = value;                                                 // пишем значение в выделенный слот
            __threadfence();                                                   // гарантируем, что запись data[pos] "видна" другим SM
            ready[pos] = 1;                                                    // помечаем слот как готовый (после fence)
            return true;                                                       // операция успешна
        }
        atomicSub(&top, 1);                                                    // если переполнение — откатываем top
        return false;                                                          // операция неуспешна
    }

    __device__ bool pop(int* value) {                                          // операция pop на GPU
        int pos = atomicSub(&top, 1) - 1;                                      // атомарно уменьшаем top и получаем индекс последнего элемента
        if (pos >= 0) {                                                        // если индекс корректный (стек был не пуст)
            while (atomicAdd(&ready[pos], 0) == 0) {                           // ждём, пока push пометит слот готовым (атомарное чтение)
            }
            *value = data[pos];                                                // читаем значение
            return true;                                                       // успешно
        }
        atomicAdd(&top, 1);                                                    // если стек был пуст — возвращаем top обратно
        return false;                                                          // неуспешно
    }
};

// ============================================================
// Параллельная очередь (FIFO) с ready[] (однопроходная, без кольца)
// ============================================================
struct Queue {                                                                 // структура очереди (на GPU)
    int* data;                                                                 // массив данных
    int* ready;                                                                // массив флагов готовности
    int  head;                                                                 // индекс чтения (dequeue)
    int  tail;                                                                 // индекс записи (enqueue)
    int  capacity;                                                             // максимальная ёмкость

    __device__ void init(int* buffer, int* readyBuf, int size) {               // инициализация на GPU
        data = buffer;                                                         // адрес данных
        ready = readyBuf;                                                      // адрес флагов
        head = 0;                                                              // начало чтения
        tail = 0;                                                              // начало записи
        capacity = size;                                                       // ёмкость
    }

    __device__ bool enqueue(int value) {                                       // добавить элемент в конец
        int pos = atomicAdd(&tail, 1);                                         // резервируем позицию и увеличиваем tail
        if (pos < capacity) {                                                  // если есть место
            data[pos] = value;                                                 // пишем значение
            __threadfence();                                                   // публикуем запись
            ready[pos] = 1;                                                    // отмечаем готовность
            return true;                                                       // успешно
        }
        atomicSub(&tail, 1);                                                   // если переполнение — откатываем tail
        return false;                                                          // неуспешно
    }

    __device__ bool dequeue(int* value) {                                      // извлечь элемент из начала
        int pos = atomicAdd(&head, 1);                                         // резервируем позицию чтения и увеличиваем head
        int t   = atomicAdd(&tail, 0);                                         // атомарно читаем tail (сколько реально записывали)
        if (pos < t) {                                                         // если действительно есть элемент
            while (atomicAdd(&ready[pos], 0) == 0) {                           // ждём, пока enqueue заполнит слот
            }
            *value = data[pos];                                                // читаем значение
            return true;                                                       // успешно
        }
        atomicSub(&head, 1);                                                   // если очереди не было — откатываем head
        return false;                                                          // неуспешно
    }
};

// ============================================================
// Kernel: обнулить int-массив (например ready[])
// ============================================================
__global__ void fillZeroKernel(int* a, int n) {                                 // kernel для обнуления
    int i = blockIdx.x * blockDim.x + threadIdx.x;                              // глобальный индекс потока
    if (i < n) a[i] = 0;                                                       // если в диапазоне — записываем 0
}

// ============================================================
// Kernel: инициализация стека (1 поток)
// ============================================================
__global__ void initStackKernel(Stack* s, int* buffer, int* readyBuf, int cap) {// kernel инициализации стека
    if (threadIdx.x == 0 && blockIdx.x == 0) {                                  // только один поток выполняет init
        s->init(buffer, readyBuf, cap);                                         // вызываем device-метод init
    }
}

// ============================================================
// Kernel: инициализация очереди (1 поток)
// ============================================================
__global__ void initQueueKernel(Queue* q, int* buffer, int* readyBuf, int cap) {// kernel инициализации очереди
    if (threadIdx.x == 0 && blockIdx.x == 0) {                                  // только один поток
        q->init(buffer, readyBuf, cap);                                         // init
    }
}

// ============================================================
// Kernels: параллельные операции со стеком
// ============================================================
__global__ void stackPushKernel(Stack* s, int nOps, int* ok) {                  // kernel: много push
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                             // глобальный id потока
    if (tid < nOps) {                                                          // если поток участвует в операции
        ok[tid] = s->push(tid) ? 1 : 0;                                         // пытаемся push(tid), записываем 1/0
    }
}

__global__ void stackPopKernel(Stack* s, int nOps, int* out, int* ok) {         // kernel: много pop
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                             // id потока
    if (tid < nOps) {                                                          // проверка границ
        int v = -1;                                                            // локальная переменная для результата
        bool success = s->pop(&v);                                             // пытаемся pop
        ok[tid] = success ? 1 : 0;                                             // успех/неуспех
        out[tid] = v;                                                          // записываем извлечённое значение (или -1)
    }
}

// ============================================================
// Kernels: параллельные операции с очередью
// ============================================================
__global__ void queueEnqKernel(Queue* q, int nOps, int* ok) {                   // kernel: много enqueue
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                             // id потока
    if (tid < nOps) {                                                          // если tid в диапазоне
        ok[tid] = q->enqueue(tid) ? 1 : 0;                                      // enqueue(tid) и 1/0 в ok
    }
}

__global__ void queueDeqKernel(Queue* q, int nOps, int* out, int* ok) {         // kernel: много dequeue
    int tid = blockIdx.x * blockDim.x + threadIdx.x;                             // id потока
    if (tid < nOps) {                                                          // граница
        int v = -1;                                                            // локальная переменная
        bool success = q->dequeue(&v);                                          // dequeue
        ok[tid] = success ? 1 : 0;                                              // успех
        out[tid] = v;                                                          // значение
    }
}

// ============================================================
// main(): тест корректности + исследование производительности
// ============================================================
int main() {                                                                    // точка входа программы
    const int N   = 1 << 20;                                                    // число операций (1,048,576)
    const int CAP = N;                                                          // ёмкость структур (берём ровно N)
    const int TPB = 256;                                                        // threads per block (потоков в блоке)
    const int blocks = (N + TPB - 1) / TPB;                                     // число блоков для покрытия N потоков

    Stack* d_stack = nullptr;                                                   // указатель на Stack в памяти GPU
    Queue* d_queue = nullptr;                                                   // указатель на Queue в памяти GPU

    int *d_stackData = nullptr;                                                 // буфер данных стека на GPU
    int *d_stackReady = nullptr;                                                // буфер ready стека на GPU
    int *d_queueData = nullptr;                                                 // буфер данных очереди на GPU
    int *d_queueReady = nullptr;                                                // буфер ready очереди на GPU

    int *d_ok = nullptr;                                                        // массив успешности операций на GPU
    int *d_out = nullptr;                                                       // массив результатов pop/dequeue на GPU

    CUDA_CHECK(cudaMalloc(&d_stack, sizeof(Stack)));                            // выделяем память под Stack на GPU
    CUDA_CHECK(cudaMalloc(&d_queue, sizeof(Queue)));                            // выделяем память под Queue на GPU

    CUDA_CHECK(cudaMalloc(&d_stackData,  CAP * sizeof(int)));                   // память под данные стека
    CUDA_CHECK(cudaMalloc(&d_stackReady, CAP * sizeof(int)));                   // память под ready стека

    CUDA_CHECK(cudaMalloc(&d_queueData,  CAP * sizeof(int)));                   // память под данные очереди
    CUDA_CHECK(cudaMalloc(&d_queueReady, CAP * sizeof(int)));                   // память под ready очереди

    CUDA_CHECK(cudaMalloc(&d_ok,  N * sizeof(int)));                            // память под ok (N элементов)
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(int)));                            // память под out (N элементов)

    std::vector<int> ok(N);                                                     // CPU-вектор для ok
    std::vector<int> out(N);                                                    // CPU-вектор для out

    // -------------------- ТЕСТ КОРРЕКТНОСТИ: STACK --------------------
    initStackKernel<<<1, 1>>>(d_stack, d_stackData, d_stackReady, CAP);         // инициализируем стек (1 блок, 1 поток)
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём завершения инициализации

    fillZeroKernel<<<blocks, TPB>>>(d_stackReady, CAP);                         // обнуляем ready[] у стека
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём обнуления

    stackPushKernel<<<blocks, TPB>>>(d_stack, N, d_ok);                         // параллельно делаем N push
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём завершения push

    CUDA_CHECK(cudaMemcpy(ok.data(), d_ok, N * sizeof(int), cudaMemcpyDeviceToHost)); // копируем ok на CPU
    std::cout << "Stack push success: " << sum01(ok) << "/" << N << "\n";       // печатаем успех push

    stackPopKernel<<<blocks, TPB>>>(d_stack, N, d_out, d_ok);                   // параллельно делаем N pop
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём завершения pop

    CUDA_CHECK(cudaMemcpy(ok.data(), d_ok,  N * sizeof(int), cudaMemcpyDeviceToHost)); // ok на CPU
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost)); // out на CPU
    std::cout << "Stack pop  success: " << sum01(ok) << "/" << N << "\n";       // печатаем успех pop
    std::cout << "Stack permutation ok: " << (checkPermutation(out, N) ? "YES" : "NO") << "\n"; // проверка перестановки

    // -------------------- ТЕСТ КОРРЕКТНОСТИ: QUEUE --------------------
    initQueueKernel<<<1, 1>>>(d_queue, d_queueData, d_queueReady, CAP);         // инициализируем очередь
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём

    fillZeroKernel<<<blocks, TPB>>>(d_queueReady, CAP);                         // обнуляем ready[] у очереди
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём

    queueEnqKernel<<<blocks, TPB>>>(d_queue, N, d_ok);                          // параллельно делаем N enqueue
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём

    CUDA_CHECK(cudaMemcpy(ok.data(), d_ok, N * sizeof(int), cudaMemcpyDeviceToHost)); // ok на CPU
    std::cout << "Queue enq success: " << sum01(ok) << "/" << N << "\n";        // печатаем успех enqueue

    queueDeqKernel<<<blocks, TPB>>>(d_queue, N, d_out, d_ok);                   // параллельно делаем N dequeue
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём

    CUDA_CHECK(cudaMemcpy(ok.data(), d_ok,  N * sizeof(int), cudaMemcpyDeviceToHost)); // ok на CPU
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, N * sizeof(int), cudaMemcpyDeviceToHost)); // out на CPU
    std::cout << "Queue deq success: " << sum01(ok) << "/" << N << "\n";        // успех dequeue
    std::cout << "Queue permutation ok: " << (checkPermutation(out, N) ? "YES" : "NO") << "\n"; // проверка

    // -------------------- ИССЛЕДОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ --------------------
    // Идея: измеряем отдельно push/pop и enqueue/dequeue.
    // ВАЖНО: перед pop/dequeue структура должна быть заполнена.

    // ---- Stack push time ----
    initStackKernel<<<1, 1>>>(d_stack, d_stackData, d_stackReady, CAP);         // заново обнуляем top внутри стека
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём
    fillZeroKernel<<<blocks, TPB>>>(d_stackReady, CAP);                         // ready=0
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём

    float tStackPush = timeKernel([&]() {                                       // замеряем среднее время stackPushKernel
        stackPushKernel<<<blocks, TPB>>>(d_stack, N, d_ok);                     // запуск kernel push
    });

    // ---- Stack pop time (сначала заполняем стек) ----
    initStackKernel<<<1, 1>>>(d_stack, d_stackData, d_stackReady, CAP);         // обнуляем стек
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём
    fillZeroKernel<<<blocks, TPB>>>(d_stackReady, CAP);                         // ready=0
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём
    stackPushKernel<<<blocks, TPB>>>(d_stack, N, d_ok);                         // заполняем стек N элементами
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём заполнения

    float tStackPop = timeKernel([&]() {                                        // замеряем pop
        stackPopKernel<<<blocks, TPB>>>(d_stack, N, d_out, d_ok);               // запуск kernel pop
    });

    // ---- Queue enqueue time ----
    initQueueKernel<<<1, 1>>>(d_queue, d_queueData, d_queueReady, CAP);         // обнуляем очередь (head=0, tail=0)
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём
    fillZeroKernel<<<blocks, TPB>>>(d_queueReady, CAP);                         // ready=0
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём

    float tQueueEnq = timeKernel([&]() {                                        // замер enqueue
        queueEnqKernel<<<blocks, TPB>>>(d_queue, N, d_ok);                      // запуск enqueue
    });

    // ---- Queue dequeue time (сначала заполняем очередь) ----
    initQueueKernel<<<1, 1>>>(d_queue, d_queueData, d_queueReady, CAP);         // обнуляем
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём
    fillZeroKernel<<<blocks, TPB>>>(d_queueReady, CAP);                         // ready=0
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём
    queueEnqKernel<<<blocks, TPB>>>(d_queue, N, d_ok);                          // заполняем очередь
    CUDA_CHECK(cudaDeviceSynchronize());                                        // ждём

    float tQueueDeq = timeKernel([&]() {                                        // замер dequeue
        queueDeqKernel<<<blocks, TPB>>>(d_queue, N, d_out, d_ok);               // запуск dequeue
    });

    // ---- Печать результатов (в миллисекундах) ----
    std::cout << "Avg Stack push kernel time (ms): " << tStackPush << "\n";     // среднее время push
    std::cout << "Avg Stack pop  kernel time (ms): " << tStackPop  << "\n";     // среднее время pop
    std::cout << "Avg Queue enq  kernel time (ms): " << tQueueEnq  << "\n";     // среднее время enqueue
    std::cout << "Avg Queue deq  kernel time (ms): " << tQueueDeq  << "\n";     // среднее время dequeue

    // -------------------- Освобождение памяти --------------------
    CUDA_CHECK(cudaFree(d_out));                                                // освобождаем d_out
    CUDA_CHECK(cudaFree(d_ok));                                                 // освобождаем d_ok
    CUDA_CHECK(cudaFree(d_queueReady));                                         // освобождаем ready очереди
    CUDA_CHECK(cudaFree(d_queueData));                                          // освобождаем data очереди
    CUDA_CHECK(cudaFree(d_stackReady));                                         // освобождаем ready стека
    CUDA_CHECK(cudaFree(d_stackData));                                          // освобождаем data стека
    CUDA_CHECK(cudaFree(d_queue));                                              // освобождаем структуру Queue
    CUDA_CHECK(cudaFree(d_stack));                                              // освобождаем структуру Stack

    return 0;                                                                   // успешное завершение
}
