#include <cuda_runtime.h>                          // базовые функции CUDA (malloc/copy/ошибки)
#include <device_launch_parameters.h>              // параметры запуска kernel (blockIdx, threadIdx)

#include <iostream>                                // std::cout, std::cerr
#include <vector>                                  // std::vector
#include <random>                                  // генератор случайных чисел
#include <cstdint>                                 // фиксированные типы (int32_t и т.п.)
#include <cmath>                                   // мат. функции (на будущее)

#define CUDA_CHECK(call) do {                      /* макрос: оборачивает вызовы CUDA */ \
    cudaError_t err = (call);                      /* выполняем call и получаем код ошибки */ \
    if (err != cudaSuccess) {                      /* если ошибка есть */ \
        std::cerr << "CUDA error: "                /* печатаем префикс */ \
                  << cudaGetErrorString(err)       /* строковое описание ошибки */ \
                  << " at " << __FILE__            /* имя файла */ \
                  << ":" << __LINE__               /* номер строки */ \
                  << std::endl;                    /* перевод строки */ \
        std::exit(1);                              /* аварийный выход */ \
    }                                              /* конец if */ \
} while(0)                                         /* do-while чтобы макрос был как одна инструкция */

static inline float elapsed_ms(cudaEvent_t a, cudaEvent_t b) { // функция: время между событиями CUDA
    float ms = 0.0f;                            // сюда запишем миллисекунды
    CUDA_CHECK(cudaEventElapsedTime(&ms, a, b)); // считаем разницу времени между событиями
    return ms;                                  // возвращаем измеренное время
}                                               // конец функции elapsed_ms

// ---------------------------                                                     // раздел комментариев
// РЕДУКЦИЯ A: GLOBAL-ONLY                                                         // вариант A
// ---------------------------                                                     // раздел комментариев
__global__ void reduce_global_only_stage(                                          // kernel: редукция только через global
    const float* __restrict__ in,                                                  // входной массив в global памяти
    float* __restrict__ g_work,                                                    // рабочий буфер в global памяти (grid*block)
    float* __restrict__ out_block,                                                 // выход: суммы по блокам (grid)
    int n                                                                          // размер входного массива
) {                                                                                // начало kernel
    int tid = threadIdx.x;                                                         // индекс потока внутри блока
    int block = blockIdx.x;                                                        // индекс блока
    int idx = block * blockDim.x + tid;                                            // глобальный индекс элемента входа

    float x = (idx < n) ? in[idx] : 0.0f;                                          // читаем элемент или 0, если вышли за границу

    int work_idx = block * blockDim.x + tid;                                       // индекс в рабочем буфере (global)
    g_work[work_idx] = x;                                                          // кладём значение в global рабочий буфер

    __syncthreads();                                                               // ждём, пока все потоки запишут свои данные

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {                  // редукция: шаг уменьшается в 2 раза
        if (tid < stride) {                                                        // участвует только первая половина потоков
            g_work[work_idx] += g_work[work_idx + stride];                         // складываем пары прямо в global памяти
        }                                                                          // конец if
        __syncthreads();                                                           // синхронизация перед следующим шагом
    }                                                                              // конец цикла for

    if (tid == 0) {                                                                // только поток 0 в блоке
        out_block[block] = g_work[block * blockDim.x];                             // записывает сумму блока в компактный массив
    }                                                                              // конец if
}                                                                                  // конец kernel reduce_global_only_stage

// ---------------------------                                                     // раздел комментариев
// РЕДУКЦИЯ B: GLOBAL + SHARED                                                     // вариант B
// ---------------------------                                                     // раздел комментариев
__global__ void reduce_shared_stage(                                               // kernel: редукция с shared memory
    const float* __restrict__ in,                                                  // входной массив в global памяти
    float* __restrict__ out_block,                                                 // выход: суммы по блокам
    int n                                                                          // размер входа
) {                                                                                // начало kernel
    extern __shared__ float s[];                                                   // динамическая shared память (размер = threads*sizeof(float))
    int tid = threadIdx.x;                                                         // индекс потока внутри блока
    int idx = blockIdx.x * blockDim.x + tid;                                       // глобальный индекс элемента

    s[tid] = (idx < n) ? in[idx] : 0.0f;                                           // загрузка global -> shared (или 0)
    __syncthreads();                                                               // синхронизация после загрузки

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {                  // редукция в shared
        if (tid < stride) s[tid] += s[tid + stride];                               // складываем элементы в shared
        __syncthreads();                                                           // синхронизация на каждом шаге
    }                                                                              // конец цикла for

    if (tid == 0) out_block[blockIdx.x] = s[0];                                    // поток 0 записывает сумму блока в global
}                                                                                  // конец kernel reduce_shared_stage

// ---------------------------                                                     // раздел комментариев
// Плиточная сортировка: local bubble + merge in shared                             // сортировка по плиткам
// ---------------------------                                                     // раздел комментариев
constexpr int BLOCK = 256;                                                         // потоков в блоке (и размер блока для kernels)
constexpr int ELEMS_PER_THREAD = 4;                                                // сколько элементов сортирует каждый поток локально
constexpr int TILE = BLOCK * ELEMS_PER_THREAD;                                     // размер плитки (элементов на блок сортировки)

__device__ inline void bubble_sort_local(float a[ELEMS_PER_THREAD]) {              // device-функция: пузырёк в локальном массиве
    #pragma unroll                                                                 // просим компилятор развернуть цикл
    for (int i = 0; i < ELEMS_PER_THREAD; i++) {                                   // внешний цикл пузырька
        #pragma unroll                                                             // развернуть внутренний цикл
        for (int j = 0; j < ELEMS_PER_THREAD - 1 - i; j++) {                       // сравниваем соседей
            if (a[j] > a[j + 1]) {                                                 // если порядок неправильный
                float t = a[j];                                                    // временная переменная
                a[j] = a[j + 1];                                                   // меняем местами
                a[j + 1] = t;                                                      // завершаем swap
            }                                                                      // конец if
        }                                                                          // конец внутреннего for
    }                                                                              // конец внешнего for
}                                                                                  // конец bubble_sort_local

__device__ void merge_shared(float* s, float* tmp, int l, int m, int r) {          // device-функция: слияние 2 отсортированных отрезков
    int i = l, j = m, k = l;                                                       // i по левому, j по правому, k по выходу
    while (i < m && j < r)                                                         // пока оба отрезка не закончились
        tmp[k++] = (s[i] <= s[j]) ? s[i++] : s[j++];                               // выбираем меньший и пишем в tmp
    while (i < m) tmp[k++] = s[i++];                                               // дописываем хвост левого
    while (j < r) tmp[k++] = s[j++];                                               // дописываем хвост правого
    for (int p = l; p < r; p++) s[p] = tmp[p];                                     // копируем результат назад в s
}                                                                                  // конец merge_shared

__global__ void tile_sort_local_and_merge(float* d_data, int n)                    // kernel: сортировка одной плитки на блок
{                                                                                  // начало kernel
    __shared__ float s[TILE];                                                      // shared: данные плитки (размер TILE)
    __shared__ float tmp[TILE];                                                    // shared: временный буфер для слияния

    int tid = threadIdx.x;                                                         // индекс потока
    int tile_start = blockIdx.x * TILE;                                            // начало плитки в общем массиве

    float local[ELEMS_PER_THREAD];                                                 // локальный массив потока (регистры/локальная память)
    #pragma unroll                                                                 // развернуть цикл загрузки
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {                                   // каждый поток берёт ELEMS_PER_THREAD элементов
        int idx = tile_start + tid + k * BLOCK;                                    // индекс элемента: разнесение по потокам
        local[k] = (idx < n) ? d_data[idx] : 1e30f;                                // читаем или кладём "бесконечность" для хвоста
    }                                                                              // конец for загрузки

    bubble_sort_local(local);                                                      // сортируем локальные 4 элемента пузырьком

    #pragma unroll                                                                 // развернуть цикл записи
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {                                   // записываем локально-отсортированные куски
        s[tid + k * BLOCK] = local[k];                                             // кладём в shared, формируя много маленьких "ранов"
    }                                                                              // конец for записи
    __syncthreads();                                                               // ждём, пока все потоки запишут в shared

    if (tid == 0) {                                                                // один поток выполняет последовательное слияние
        int run = ELEMS_PER_THREAD;                                                // начальная длина "рана" = 4
        while (run < TILE) {                                                       // пока не сольём всё в один отсортированный массив
            for (int start = 0; start < TILE; start += 2 * run) {                  // берём пары отрезков длиной run
                int l = start;                                                     // левая граница
                int m = min(start + run, TILE);                                    // середина (граница между 2 отрезками)
                int r = min(start + 2 * run, TILE);                                // правая граница
                merge_shared(s, tmp, l, m, r);                                      // сливаем два отрезка в shared
            }                                                                      // конец for по start
            run <<= 1;                                                             // увеличиваем длину "рана" в 2 раза
        }                                                                          // конец while
    }                                                                              // конец if tid==0
    __syncthreads();                                                               // ждём завершения слияния, чтобы все видели отсортированную плитку

    for (int k = tid; k < TILE; k += BLOCK) {                                      // каждый поток пишет часть плитки обратно
        int idx = tile_start + k;                                                  // глобальный индекс элемента плитки
        if (idx < n) d_data[idx] = s[k];                                           // запись shared -> global (если в пределах массива)
    }                                                                              // конец for
}                                                                                  // конец kernel tile_sort_local_and_merge

// ---------------------------                                                     // раздел комментариев
// Буферы для многошаговой редукции                                                 // храним промежуточные массивы
// ---------------------------                                                     // раздел комментариев
struct ReduceBuffers {                                                             // структура для управления GPU-буферами
    float* d_buf1 = nullptr;                                                       // первый промежуточный буфер (GPU)
    float* d_buf2 = nullptr;                                                       // второй промежуточный буфер (GPU)
    float* d_work = nullptr;                                                       // рабочий буфер для global-only (GPU)
    int cap1 = 0, cap2 = 0, capWork = 0;                                           // текущие ёмкости буферов (в элементах)

    void ensure(int n, bool needWork, int blocks, int threads) {                   // гарантирует, что памяти хватает
        if (n > cap1) {                                                            // если d_buf1 маловат
            if (d_buf1) CUDA_CHECK(cudaFree(d_buf1));                              // освобождаем старый
            CUDA_CHECK(cudaMalloc(&d_buf1, n * sizeof(float)));                    // выделяем новый
            cap1 = n;                                                              // обновляем ёмкость
        }                                                                          // конец if
        if (n > cap2) {                                                            // если d_buf2 маловат
            if (d_buf2) CUDA_CHECK(cudaFree(d_buf2));                              // освобождаем старый
            CUDA_CHECK(cudaMalloc(&d_buf2, n * sizeof(float)));                    // выделяем новый
            cap2 = n;                                                              // обновляем ёмкость
        }                                                                          // конец if
        if (needWork) {                                                            // если нужен рабочий буфер (global-only)
            int need = blocks * threads;                                           // сколько элементов нужно под g_work
            if (need > capWork) {                                                  // если текущего не хватает
                if (d_work) CUDA_CHECK(cudaFree(d_work));                          // освобождаем старый
                CUDA_CHECK(cudaMalloc(&d_work, (size_t)need * sizeof(float)));     // выделяем новый
                capWork = need;                                                    // обновляем ёмкость
            }                                                                      // конец if
        }                                                                          // конец if needWork
    }                                                                              // конец ensure

    ~ReduceBuffers() {                                                             // деструктор: освобождение памяти
        if (d_buf1) cudaFree(d_buf1);                                              // освобождаем buf1
        if (d_buf2) cudaFree(d_buf2);                                              // освобождаем buf2
        if (d_work) cudaFree(d_work);                                              // освобождаем work
    }                                                                              // конец деструктора
};                                                                                 // конец struct

void reduce_global_only_full(const float* d_in, int n, float* d_out1,              // функция: редукция A до 1 числа
                             ReduceBuffers& rb, int threads) {                    // буферы + число потоков
    int blocks = (n + threads - 1) / threads;                                      // сколько блоков нужно для покрытия n
    rb.ensure(n, true, blocks, threads);                                           // убеждаемся, что буферы выделены

    float* cur_out = rb.d_buf1;                                                    // текущий выход (после 1 стадии) в buf1

    reduce_global_only_stage<<<blocks, threads, 0>>>(d_in, rb.d_work, cur_out, n); // 1 стадия: суммы блоков
    CUDA_CHECK(cudaGetLastError());                                                // проверяем, что kernel запустился без ошибки

    int cur_n = blocks;                                                            // теперь размер массива = число блоков

    while (cur_n > 1) {                                                            // пока не сведём к одному элементу
        int b = (cur_n + threads - 1) / threads;                                   // блоки для следующей стадии
        rb.ensure(cur_n, true, b, threads);                                        // расширяем буферы при необходимости

        float* next_out = (cur_out == rb.d_buf1) ? rb.d_buf2 : rb.d_buf1;          // чередуем buf1/buf2
        reduce_global_only_stage<<<b, threads, 0>>>(cur_out, rb.d_work, next_out,  // редуцируем дальше
                                                   cur_n);                        // размер текущего входа
        CUDA_CHECK(cudaGetLastError());                                            // проверка ошибок запуска

        cur_out = next_out;                                                        // обновляем текущий буфер
        cur_n = b;                                                                 // обновляем текущий размер
    }                                                                              // конец while

    CUDA_CHECK(cudaMemcpy(d_out1, cur_out, sizeof(float), cudaMemcpyDeviceToDevice)); // копируем итог (1 float) в d_out1
}                                                                                  // конец reduce_global_only_full

void reduce_shared_full(const float* d_in, int n, float* d_out1,                   // функция: редукция B до 1 числа
                        ReduceBuffers& rb, int threads) {                          // буферы + потоки
    int blocks = (n + threads - 1) / threads;                                      // блоки для 1 стадии
    rb.ensure(n, false, blocks, threads);                                          // work не нужен, поэтому needWork=false

    float* cur_out = rb.d_buf1;                                                    // текущий выход в buf1

    reduce_shared_stage<<<blocks, threads, threads * sizeof(float)>>>(d_in,        // 1 стадия: shared-редукция
                                                                     cur_out, n);  // выход и размер
    CUDA_CHECK(cudaGetLastError());                                                // проверка запуска

    int cur_n = blocks;                                                            // новый размер = число блоков

    while (cur_n > 1) {                                                            // пока не станет 1
        int b = (cur_n + threads - 1) / threads;                                   // блоки для следующей стадии
        float* next_out = (cur_out == rb.d_buf1) ? rb.d_buf2 : rb.d_buf1;          // чередуем буферы
        reduce_shared_stage<<<b, threads, threads * sizeof(float)>>>(cur_out,      // редуцируем block-суммы
                                                                     next_out,     // куда писать
                                                                     cur_n);       // размер входа
        CUDA_CHECK(cudaGetLastError());                                            // проверка запуска

        cur_out = next_out;                                                        // обновляем текущий буфер
        cur_n = b;                                                                 // обновляем текущий размер
    }                                                                              // конец while

    CUDA_CHECK(cudaMemcpy(d_out1, cur_out, sizeof(float), cudaMemcpyDeviceToDevice)); // переносим итог в d_out1
}                                                                                  // конец reduce_shared_full

template <typename F>                                                             // шаблон, чтобы принимать лямбду с любым типом
float time_kernel_chain(F&& fn, int warmup, int runs)                              // функция: меряем среднее время выполнения цепочки
{                                                                                  // начало time_kernel_chain
    cudaEvent_t a, b;                                                              // два CUDA-события (старт/стоп)
    CUDA_CHECK(cudaEventCreate(&a));                                               // создаём событие a
    CUDA_CHECK(cudaEventCreate(&b));                                               // создаём событие b

    for (int i = 0; i < warmup; i++) {                                             // прогрев: несколько запусков без измерений
        fn();                                                                      // выполняем цепочку (kernel-и)
    }                                                                              // конец warmup
    CUDA_CHECK(cudaDeviceSynchronize());                                           // ждём завершения всех запусков прогрева

    float total = 0.0f;                                                            // сумма времен (для среднего)
    for (int i = 0; i < runs; i++) {                                               // основной цикл замеров
        CUDA_CHECK(cudaEventRecord(a));                                            // ставим "старт"
        fn();                                                                      // выполняем вычисления
        CUDA_CHECK(cudaEventRecord(b));                                            // ставим "стоп"
        CUDA_CHECK(cudaEventSynchronize(b));                                       // ждём, пока событие b завершится
        total += elapsed_ms(a, b);                                                 // добавляем время этого прогона
    }                                                                              // конец runs

    CUDA_CHECK(cudaEventDestroy(a));                                               // уничтожаем событие a
    CUDA_CHECK(cudaEventDestroy(b));                                               // уничтожаем событие b
    return total / runs;                                                           // возвращаем среднее время
}                                                                                  // конец time_kernel_chain

int main() {                                                                       // точка входа программы
    std::ios::sync_with_stdio(false);                                              // ускоряем ввод/вывод (не обязательно, но полезно)

    std::vector<int> sizes = {10000, 100000, 1000000};                             // размеры массивов из задания

    const int WARMUP = 5;                                                          // сколько прогревочных прогонов
    const int RUNS   = 50;                                                         // сколько прогонов для усреднения

    std::mt19937 rng(123);                                                         // генератор псевдослучайных чисел (фиксированный seed)
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);                        // распределение: числа от 0 до 1

    std::cout << "mode,n,time_ms,checksum\n";                                      // заголовок CSV для результатов

    ReduceBuffers rb;                                                              // создаём менеджер буферов для редукции
    const int threads = 256;                                                       // число потоков в блоке для редукций

    for (int n : sizes) {                                                          // цикл по размерам массивов
        std::vector<float> h(n);                                                   // массив на CPU
        for (int i = 0; i < n; i++) h[i] = dist(rng);                              // заполняем CPU-массив случайными значениями

        float* d_in = nullptr;                                                     // указатель на массив в GPU памяти
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));                          // выделяем память на GPU под входной массив
        CUDA_CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float),                   // копируем данные CPU -> GPU
                              cudaMemcpyHostToDevice));                            // направление копирования

        float* d_sum = nullptr;                                                    // указатель на 1 float на GPU для результата суммы
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));                             // выделяем 1 float на GPU

        float tA = time_kernel_chain([&]() {                                       // измеряем среднее время для варианта A
            reduce_global_only_full(d_in, n, d_sum, rb, threads);                  // выполняем полную редукцию A до 1 числа
        }, WARMUP, RUNS);                                                          // параметры прогрева и количества прогонов

        float sumA = 0.0f;                                                         // сюда скопируем сумму A на CPU
        CUDA_CHECK(cudaMemcpy(&sumA, d_sum, sizeof(float),                         // копируем 1 float GPU -> CPU
                              cudaMemcpyDeviceToHost));                            // направление копирования
        std::cout << "reduce_global_only_full," << n << "," << tA                  // печатаем строку CSV для A
                  << "," << sumA << "\n";                                          // добавляем checksum (сама сумма)

        float tB = time_kernel_chain([&]() {                                       // измеряем среднее время для варианта B
            reduce_shared_full(d_in, n, d_sum, rb, threads);                       // выполняем полную редукцию B до 1 числа
        }, WARMUP, RUNS);                                                          // параметры замера

        float sumB = 0.0f;                                                         // сюда скопируем сумму B на CPU
        CUDA_CHECK(cudaMemcpy(&sumB, d_sum, sizeof(float),                         // копируем результат B GPU -> CPU
                              cudaMemcpyDeviceToHost));                            // направление копирования
        std::cout << "reduce_global_shared_full," << n << "," << tB                // печатаем строку CSV для B
                  << "," << sumB << "\n";                                          // добавляем checksum

        float* d_sort = nullptr;                                                   // GPU-массив для сортировки (копия входа)
        CUDA_CHECK(cudaMalloc(&d_sort, n * sizeof(float)));                        // выделяем память под сортируемый массив

        int sort_blocks = (n + TILE - 1) / TILE;                                   // сколько блоков нужно для сортировки плиток

        float tS = time_kernel_chain([&]() {                                       // измеряем среднее время сортировки
            CUDA_CHECK(cudaMemcpy(d_sort, d_in, n * sizeof(float),                 // копируем вход в сортируемый буфер (GPU->GPU)
                                  cudaMemcpyDeviceToDevice));                      // направление: device->device
            tile_sort_local_and_merge<<<sort_blocks, BLOCK>>>(d_sort, n);          // запускаем сортировку плиток
            CUDA_CHECK(cudaGetLastError());                                        // проверяем ошибки запуска сортировки
        }, WARMUP, RUNS);                                                          // замер с прогревом и усреднением

        float first = 0.0f;                                                        // для checksum сортировки возьмём 1 элемент
        CUDA_CHECK(cudaMemcpy(&first, d_sort, sizeof(float),                       // читаем первый элемент массива после сортировки
                              cudaMemcpyDeviceToHost));                            // направление: device->host
        std::cout << "tile_sort_local_shared_merge," << n << "," << tS             // печатаем строку CSV для сортировки
                  << "," << first << "\n";                                         // добавляем checksum (первый элемент)

        CUDA_CHECK(cudaFree(d_sort));                                              // освобождаем буфер сортировки на GPU
        CUDA_CHECK(cudaFree(d_sum));                                               // освобождаем буфер суммы (1 float) на GPU
        CUDA_CHECK(cudaFree(d_in));                                                // освобождаем входной массив на GPU
    }                                                                              // конец цикла по sizes

    return 0;                                                                      // успешное завершение программы
}                                                                                  // конец main
