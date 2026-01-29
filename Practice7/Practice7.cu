// ===============================
// Практическая работа №7 (CUDA)
// Редукция (sum) + Сканирование (prefix sum) + Анализ производительности
// ===============================

// ---------- [Подключение CUDA заголовков] ----------
#include <cuda_runtime.h>                 // CUDA runtime API (cudaMalloc, cudaMemcpy, cudaEvent и т.д.)
#include <device_launch_parameters.h>     // Макросы/типы для kernel launch (threadIdx, blockIdx ...)

// ---------- [Подключение стандартных библиотек C++] ----------
#include <iostream>       // std::cout, std::cerr
#include <vector>         // std::vector
#include <random>         // генерация случайных чисел
#include <chrono>         // измерение времени на CPU
#include <fstream>        // запись CSV файла
#include <iomanip>        // форматированный вывод (setprecision)
#include <cmath>          // fabs, max
#include <cstring>        // memcpy

// ---------- [Макрос проверки ошибок CUDA] ----------
// CUDA_CHECK(cudaMalloc(...)) -> если ошибка, выводим сообщение и завершаем программу
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \                         /* выполняем CUDA-вызов и сохраняем код ошибки */ \
    if (err != cudaSuccess) { \                         /* если ошибка != success */ \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \                                 /* аварийное завершение */ \
    } \
} while(0)

// ---------- [Размер блока CUDA] ----------
// BLOCK = количество потоков в блоке (обычно 128/256/512)
// Это важно для производительности (доп. задание про влияние блока)
static constexpr int BLOCK = 256;

// ============================================================================
// ============================= CPU-РЕАЛИЗАЦИИ ==============================
// ============================================================================

// ----------------------------- CPU reduce (sum) -----------------------------
// Суммирование массива на CPU (нужно для сравнения с GPU)  [Задание 3: сравнение CPU/GPU]
double cpu_reduce_sum(const float* x, int n) {          // x - массив, n - размер
    double s = 0.0;                                     // сумма в double (точнее, чем float)
    for (int i = 0; i < n; ++i) s += x[i];             // последовательно прибавляем элементы
    return s;                                           // возвращаем сумму
}

// --------------------------- CPU scan (inclusive) ---------------------------
// Prefix sum на CPU: out[i] = x[0]+...+x[i] [Задание 3: сравнение CPU/GPU]
void cpu_scan_inclusive(const float* x, float* out, int n) { // вход x, выход out
    double acc = 0.0;                                   // аккумулятор суммы
    for (int i = 0; i < n; ++i) {                       // идём по массиву
        acc += x[i];                                    // добавляем текущий элемент
        out[i] = (float)acc;                            // записываем префиксную сумму
    }
}

// ============================================================================
// ======================= ЗАДАНИЕ 1: GPU РЕДУКЦИЯ (SUM) ======================
// ============================================================================

// -------------------- Вариант 1: "простая" редукция --------------------
// Здесь используется shared memory для редукции внутри блока, но без warp-оптимизаций
// В каждом блоке обрабатывается 2*BLOCK элементов -> выдаём 1 частичную сумму
__global__ void reduce_sum_global_naive(const float* __restrict__ x, // входной массив в global памяти
                                       float* __restrict__ partial, // выход: частичные суммы по блокам
                                       int n)                        // размер массива
{
    int tid = threadIdx.x;                                           // локальный индекс потока в блоке
    int i = blockIdx.x * blockDim.x * 2 + tid;                       // индекс элемента, который читает поток

    float a = (i < n) ? x[i] : 0.0f;                                 // читаем 1-й элемент (если вышли за n -> 0)
    float b = (i + blockDim.x < n) ? x[i + blockDim.x] : 0.0f;       // читаем 2-й элемент (через blockDim)
    float sum = a + b;                                               // суммируем два значения

    __shared__ float sdata[BLOCK];                                   // shared память для редукции внутри блока
    sdata[tid] = sum;                                                // каждый поток кладёт своё значение в shared
    __syncthreads();                                                 // ждём пока все потоки запишут

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {                   // редукция: шаг делим пополам
        if (tid < s) sdata[tid] += sdata[tid + s];                   // первые s потоков суммируют пары
        __syncthreads();                                             // синхронизация после каждого шага
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];                    // поток 0 записывает сумму блока
}

// -------------------- Вариант 2: редукция через shared + меньше sync --------------------
// То же самое, но оптимальнее: уменьшаем синхронизации, делаем финал через warp-synchronous
__global__ void reduce_sum_shared(const float* __restrict__ x,        // вход
                                 float* __restrict__ partial,        // выход частичных сумм
                                 int n)                               // размер
{
    int tid = threadIdx.x;                                           // индекс потока
    int base = blockIdx.x * blockDim.x * 2;                          // начало диапазона для этого блока

    int i1 = base + tid;                                             // 1-й индекс
    int i2 = base + tid + blockDim.x;                                // 2-й индекс

    float v = 0.0f;                                                  // локальная сумма потока
    if (i1 < n) v += x[i1];                                          // добавляем элемент 1
    if (i2 < n) v += x[i2];                                          // добавляем элемент 2

    __shared__ float sdata[BLOCK];                                   // shared массив
    sdata[tid] = v;                                                  // кладём значение в shared
    __syncthreads();                                                 // синхронизация

    for (int s = blockDim.x / 2; s > 32; s >>= 1) {                  // редуцируем пока больше 32
        if (tid < s) sdata[tid] += sdata[tid + s];                   // суммируем пары
        __syncthreads();                                             // синхронизация (нужна пока > 32)
    }

    // Финальные 32 элемента обрабатываем без __syncthreads() (внутри warp)
    if (tid < 32) {                                                  // только первый warp
        volatile float* smem = sdata;                                // volatile чтобы компилятор не переставлял чтения
        smem[tid] += smem[tid + 32];                                 // шаг 32
        smem[tid] += smem[tid + 16];                                 // шаг 16
        smem[tid] += smem[tid +  8];                                 // шаг 8
        smem[tid] += smem[tid +  4];                                 // шаг 4
        smem[tid] += smem[tid +  2];                                 // шаг 2
        smem[tid] += smem[tid +  1];                                 // шаг 1
    }

    if (tid == 0) partial[blockIdx.x] = sdata[0];                    // результат блока
}

// -------------------- Вариант 3: warp shuffle (самый быстрый) --------------------
// Суммируем внутри warp через регистры (__shfl_down_sync), а не через shared
__inline__ __device__ float warp_reduce_sum(float val) {             // устройство-функция (работает на GPU)
    for (int offset = 16; offset > 0; offset >>= 1)                  // половинное уменьшение offset
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);            // берём значение соседа ниже и прибавляем
    return val;                                                      // итоговая сумма внутри warp
}

__global__ void reduce_sum_warp_shuffle(const float* __restrict__ x,  // вход
                                       float* __restrict__ partial,  // выход
                                       int n)                         // размер
{
    int tid = threadIdx.x;                                           // индекс потока
    int base = blockIdx.x * blockDim.x * 2;                          // начало куска

    int i1 = base + tid;                                             // индекс 1
    int i2 = base + tid + blockDim.x;                                // индекс 2

    float v = 0.0f;                                                  // значение потока
    if (i1 < n) v += x[i1];                                          // читаем 1
    if (i2 < n) v += x[i2];                                          // читаем 2

    v = warp_reduce_sum(v);                                          // сводим сумму внутри warp

    __shared__ float warp_sums[BLOCK / 32];                          // shared для сумм warps (например 256/32=8)
    int lane = tid & 31;                                             // позиция в warp (0..31)
    int wid  = tid >> 5;                                             // номер warp в блоке (0..7)

    if (lane == 0) warp_sums[wid] = v;                               // один поток на warp пишет сумму warp в shared
    __syncthreads();                                                 // ждём пока все warps записали

    if (wid == 0) {                                                  // первый warp суммирует суммы warps
        float block_sum = (tid < (BLOCK / 32)) ? warp_sums[lane] : 0.0f; // берём warp_sums
        block_sum = warp_reduce_sum(block_sum);                      // редукция внутри warp
        if (lane == 0) partial[blockIdx.x] = block_sum;              // поток lane=0 записывает сумму блока
    }
}

// Перечисление вариантов редукции (для сравнения производительности) [Задание 3]
enum class ReduceVariant { GlobalNaive, Shared, WarpShuffle };

// -------------------- Хост-функция: редукция "до одного числа" --------------------
// Запускаем ядро много раз, пока не останется 1 значение.
// Это и есть "полная редукция" для всего массива  [Задание 1]
float gpu_reduce_sum(const float* d_in, int n, ReduceVariant variant)
{
    int cur_n = n;                                                   // текущий размер данных (уменьшается на каждом шаге)
    const float* cur_in = d_in;                                      // текущий вход (сначала d_in)

    int max_blocks = (n + (BLOCK * 2 - 1)) / (BLOCK * 2);            // максимальное число блоков на первом шаге

    float* d_buf1 = nullptr;                                         // буфер для частичных сумм (1)
    float* d_buf2 = nullptr;                                         // буфер для частичных сумм (2)
    CUDA_CHECK(cudaMalloc(&d_buf1, max_blocks * sizeof(float)));     // выделяем GPU память
    CUDA_CHECK(cudaMalloc(&d_buf2, max_blocks * sizeof(float)));     // выделяем GPU память

    bool flip = false;                                               // флаг: меняем буферы местами (ping-pong)

    while (cur_n > 1) {                                              // пока не останется 1 элемент
        int blocks = (cur_n + (BLOCK * 2 - 1)) / (BLOCK * 2);        // сколько блоков нужно
        float* d_out = flip ? d_buf2 : d_buf1;                       // куда писать результат (в один из буферов)

        if (variant == ReduceVariant::GlobalNaive)                   // если вариант 1
            reduce_sum_global_naive<<<blocks, BLOCK>>>(cur_in, d_out, cur_n); // запускаем ядро
        else if (variant == ReduceVariant::Shared)                   // вариант 2
            reduce_sum_shared<<<blocks, BLOCK>>>(cur_in, d_out, cur_n);       // запускаем ядро
        else                                                         // вариант 3
            reduce_sum_warp_shuffle<<<blocks, BLOCK>>>(cur_in, d_out, cur_n); // запускаем ядро

        CUDA_CHECK(cudaGetLastError());                               // проверяем ошибки запуска ядра

        cur_in = d_out;                                              // следующий вход = частичные суммы
        cur_n = blocks;                                              // новый размер = число блоков (частичных сумм)
        flip = !flip;                                                // меняем буфер
    }

    float result = 0.0f;                                             // итог на CPU
    CUDA_CHECK(cudaMemcpy(&result, cur_in, sizeof(float), cudaMemcpyDeviceToHost)); // копируем одно число на CPU

    cudaFree(d_buf1);                                                // освобождаем буфер 1
    cudaFree(d_buf2);                                                // освобождаем буфер 2
    return result;                                                   // возвращаем сумму
}

// ============================================================================
// ================== ЗАДАНИЕ 2: GPU СКАНИРОВАНИЕ (PREFIX SUM) =================
// ============================================================================

// Blelloch scan по блоку: делает EXCLUSIVE scan на 2*BLOCK элементов,
// а также сохраняет сумму блока (для иерархического scan)
__global__ void scan_blelloch_block_exclusive(const float* __restrict__ in, // вход
                                              float* __restrict__ out,      // выход (exclusive scan)
                                              float* __restrict__ block_sums,// сумма каждого блока
                                              int n)                         // размер
{
    __shared__ float temp[2 * BLOCK];                              // shared память на 2*BLOCK элементов

    int tid = threadIdx.x;                                         // индекс потока
    int block_start = 2 * blockIdx.x * blockDim.x;                 // начало данных для блока

    int i1 = block_start + tid;                                    // индекс 1
    int i2 = block_start + tid + blockDim.x;                       // индекс 2

    temp[tid] = (i1 < n) ? in[i1] : 0.0f;                          // загружаем 1-й элемент
    temp[tid + blockDim.x] = (i2 < n) ? in[i2] : 0.0f;             // загружаем 2-й элемент

    // -------- Up-sweep (строим дерево сумм) --------
    int offset = 1;                                                // шаг
    for (int d = blockDim.x; d > 0; d >>= 1) {                     // уменьшаем d в 2 раза
        __syncthreads();                                           // синхронизация перед этапом
        if (tid < d) {                                             // активны только первые d потоков
            int ai = offset * (2 * tid + 1) - 1;                   // индекс A
            int bi = offset * (2 * tid + 2) - 1;                   // индекс B
            temp[bi] += temp[ai];                                  // суммируем
        }
        offset <<= 1;                                              // offset *= 2
    }

    // temp[last] содержит сумму блока
    if (tid == 0) {                                                // один поток
        if (block_sums) block_sums[blockIdx.x] = temp[2 * blockDim.x - 1]; // сохраняем сумму блока
        temp[2 * blockDim.x - 1] = 0.0f;                           // делаем exclusive scan (последний = 0)
    }

    // -------- Down-sweep (распределяем префиксы) --------
    for (int d = 1; d <= blockDim.x; d <<= 1) {                    // d растёт в 2 раза
        offset >>= 1;                                              // offset /= 2
        __syncthreads();                                           // синхронизация
        if (tid < d) {                                             // активны первые d потоков
            int ai = offset * (2 * tid + 1) - 1;                   // индекс A
            int bi = offset * (2 * tid + 2) - 1;                   // индекс B
            float t = temp[ai];                                    // сохраняем A
            temp[ai] = temp[bi];                                   // A = B
            temp[bi] += t;                                         // B = B + старое A
        }
    }
    __syncthreads();                                               // финальная синхронизация

    if (i1 < n) out[i1] = temp[tid];                               // записываем 1-й результат
    if (i2 < n) out[i2] = temp[tid + blockDim.x];                  // записываем 2-й результат
}

// Добавляем смещение блока ко всем его элементам (чтобы из block scan получить общий scan)
__global__ void add_block_offsets(float* __restrict__ data,          // данные (уже просканены внутри блока)
                                  const float* __restrict__ block_offsets, // смещения для каждого блока
                                  int n)                              // размер
{
    int tid = threadIdx.x;                                          // индекс потока
    int block_start = 2 * blockIdx.x * blockDim.x;                  // начало блока в глобальных данных

    float add = block_offsets[blockIdx.x];                          // смещение для этого блока

    int i1 = block_start + tid;                                     // индекс 1
    int i2 = block_start + tid + blockDim.x;                        // индекс 2

    if (i1 < n) data[i1] += add;                                    // добавляем смещение
    if (i2 < n) data[i2] += add;                                    // добавляем смещение
}

// Делаем из INCLUSIVE scan по block_sums -> EXCLUSIVE offsets (offsets[0]=0, offsets[i]=inc[i-1])
__global__ void make_exclusive_offsets(const float* __restrict__ inc,
                                       float* __restrict__ exc,
                                       int m)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;                  // глобальный индекс
    if (i < m) exc[i] = (i == 0) ? 0.0f : inc[i - 1];               // сдвиг на 1 вправо
}

// Преобразуем EXCLUSIVE scan массива в INCLUSIVE: inclusive[i] = exclusive[i] + in[i]
__global__ void to_inclusive_kernel(const float* __restrict__ in,
                                    float* __restrict__ ex,
                                    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;                  // глобальный индекс
    if (i < n) ex[i] = ex[i] + in[i];                               // делаем inclusive
}

// Хост-функция сканирования: иерархический Blelloch scan для больших массивов [Задание 2]
void gpu_scan_inclusive(const float* d_in, float* d_out, int n)
{
    int blocks = (n + (2 * BLOCK - 1)) / (2 * BLOCK);               // количество блоков

    float* d_block_sums = nullptr;                                  // суммы блоков
    float* d_block_offsets = nullptr;                               // offsets блоков

    if (blocks > 1) {                                               // если блоков больше 1
        CUDA_CHECK(cudaMalloc(&d_block_sums, blocks * sizeof(float)));      // память под block sums
        CUDA_CHECK(cudaMalloc(&d_block_offsets, blocks * sizeof(float)));  // память под offsets
    }

    // 1) scan внутри каждого блока (exclusive) + получаем block_sums
    scan_blelloch_block_exclusive<<<blocks, BLOCK>>>(d_in, d_out, d_block_sums, n);
    CUDA_CHECK(cudaGetLastError());                                 // проверка ошибок

    if (blocks > 1) {                                               // если есть несколько блоков
        float* d_block_scan_inclusive = nullptr;                    // сюда запишем inclusive scan block_sums
        CUDA_CHECK(cudaMalloc(&d_block_scan_inclusive, blocks * sizeof(float)));

        // 2) рекурсивно сканируем массив block_sums (получаем inclusive)
        gpu_scan_inclusive(d_block_sums, d_block_scan_inclusive, blocks);

        // 3) преобразуем inclusive -> exclusive offsets
        int t = 256;                                                // threads для маленького ядра
        int b = (blocks + t - 1) / t;                               // blocks для маленького ядра
        make_exclusive_offsets<<<b, t>>>(d_block_scan_inclusive, d_block_offsets, blocks);
        CUDA_CHECK(cudaGetLastError());

        // 4) добавляем offsets ко всем элементам блоков
        add_block_offsets<<<blocks, BLOCK>>>(d_out, d_block_offsets, n);
        CUDA_CHECK(cudaGetLastError());

        cudaFree(d_block_scan_inclusive);                           // освобождаем временный буфер
    }

    // 5) сейчас d_out = exclusive scan для всего массива, переводим в inclusive
    int t2 = 256;                                                   // threads для преобразования
    int b2 = (n + t2 - 1) / t2;                                     // blocks для преобразования
    to_inclusive_kernel<<<b2, t2>>>(d_in, d_out, n);                // inclusive = exclusive + in
    CUDA_CHECK(cudaGetLastError());

    if (d_block_sums) cudaFree(d_block_sums);                       // освобождаем block_sums
    if (d_block_offsets) cudaFree(d_block_offsets);                 // освобождаем offsets
}

// ============================================================================
// ============ ЗАДАНИЕ 3: АНАЛИЗ ПРОИЗВОДИТЕЛЬНОСТИ + СРАВНЕНИЕ ===============
// ============================================================================

// ---------- GPU таймер через cudaEvent (измеряем время на GPU) ----------
float time_gpu_ms(void (*fn)(void*), void* arg) {                   // fn - функция, arg - её аргумент
    cudaEvent_t start, stop;                                        // события начала/конца
    CUDA_CHECK(cudaEventCreate(&start));                            // создаём событие start
    CUDA_CHECK(cudaEventCreate(&stop));                             // создаём событие stop
    CUDA_CHECK(cudaEventRecord(start));                             // записываем start
    fn(arg);                                                        // выполняем измеряемую функцию
    CUDA_CHECK(cudaEventRecord(stop));                              // записываем stop
    CUDA_CHECK(cudaEventSynchronize(stop));                         // ждём завершения stop
    float ms = 0.0f;                                                // здесь будет время
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));             // вычисляем время в мс
    cudaEventDestroy(start);                                        // удаляем start
    cudaEventDestroy(stop);                                         // удаляем stop
    return ms;                                                      // возвращаем время
}

// ---------- CPU таймер через chrono ----------
template <class F>
double time_cpu_ms(F&& fn) {                                        // fn - любая функция/лямбда
    auto t0 = std::chrono::high_resolution_clock::now();            // старт
    fn();                                                           // выполняем
    auto t1 = std::chrono::high_resolution_clock::now();            // конец
    std::chrono::duration<double, std::milli> dt = t1 - t0;         // разница во времени
    return dt.count();                                              // возвращаем мс
}

// ---------- Проверка "почти равны" (из-за float допускаем маленькую разницу) ----------
bool almost_equal(float a, float b, float eps = 1e-3f) {            // eps - допустимая ошибка
    float diff = std::fabs(a - b);                                  // абсолютная разница
    float norm = std::max(1.0f, std::max(std::fabs(a), std::fabs(b))); // нормализация
    return diff / norm < eps;                                       // относительная ошибка < eps
}

// ---------- Проверка корректности scan ----------
bool check_scan(const std::vector<float>& cpu, const std::vector<float>& gpu) {
    for (size_t i = 0; i < cpu.size(); ++i) {                       // сравниваем каждый элемент
        if (!almost_equal(cpu[i], gpu[i], 2e-3f)) {                 // если не равно
            std::cerr << "Mismatch scan at i=" << i
                      << " cpu=" << cpu[i] << " gpu=" << gpu[i] << "\n";
            return false;                                           // FAIL
        }
    }
    return true;                                                    // OK
}

// ---------- Обёртки, чтобы time_gpu_ms мог вызывать reduce/scan ----------
struct ReduceArgs {                                                 // аргументы для редукции
    const float* d_in;                                              // вход на GPU
    int n;                                                          // размер
    ReduceVariant v;                                                // вариант редукции
    float* out;                                                     // куда записать итоговую сумму (на CPU переменная)
};

void reduce_wrap(void* p) {                                         // функция-обёртка
    auto* a = (ReduceArgs*)p;                                       // приводим тип
    *(a->out) = gpu_reduce_sum(a->d_in, a->n, a->v);                // вызываем редукцию и пишем результат
}

struct ScanArgs {                                                   // аргументы для scan
    const float* d_in;                                              // вход на GPU
    float* d_out;                                                   // выход на GPU
    int n;                                                          // размер
};

void scan_wrap(void* p) {                                           // обёртка scan
    auto* a = (ScanArgs*)p;                                         // приводим тип
    gpu_scan_inclusive(a->d_in, a->d_out, a->n);                    // делаем scan
    CUDA_CHECK(cudaDeviceSynchronize());                            // ждём завершения (важно для корректного тайминга)
}

// ============================================================================
// ================================== MAIN ===================================
// ============================================================================

int main() {
    // Размеры из задания (1024, 1 000 000, 10 000 000)
    std::vector<int> sizes = { 1024, 1'000'000, 10'000'000 };

    // Открываем CSV-файл для записи результатов производительности [Задание 3]
    std::ofstream csv("perf.csv");
    csv << "N,cpu_reduce_ms,gpu_reduce_global_ms,gpu_reduce_shared_ms,gpu_reduce_warp_ms,cpu_scan_ms,gpu_scan_ms\n";

    // Генератор случайных чисел
    std::mt19937 rng(123);                                          // seed = 123 (чтобы результаты были повторяемыми)
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);         // числа от 0 до 1

    std::cout << "BLOCK = " << BLOCK << "\n";                       // вывод размера блока

    // Для каждого размера массива делаем тест [Задание 3]
    for (int n : sizes) {
        std::cout << "=============================\n";             // разделитель
        std::cout << "N = " << n << "\n";                           // вывод размера

        // --------- Создаём входной массив на CPU ---------
        std::vector<float> h_in(n);                                 // host input
        for (int i = 0; i < n; ++i) h_in[i] = dist(rng);            // заполняем случайными числами

        // --------- CPU reduce (для сравнения) ---------
        double cpu_sum = 0.0;                                       // здесь будет сумма CPU
        double cpu_reduce_ms = time_cpu_ms([&]() {                  // измеряем время
            cpu_sum = cpu_reduce_sum(h_in.data(), n);               // считаем сумму CPU
        });

        // --------- CPU scan (для сравнения) ---------
        std::vector<float> h_cpu_scan(n);                           // CPU scan output
        double cpu_scan_ms = time_cpu_ms([&]() {                    // измеряем время
            cpu_scan_inclusive(h_in.data(), h_cpu_scan.data(), n);  // считаем prefix sum CPU
        });

        // --------- Pinned memory (ускоряет H2D копирование) [Оптимизация памяти: pinned] ---------
        float* h_pinned = nullptr;                                  // указатель pinned памяти
        CUDA_CHECK(cudaMallocHost(&h_pinned, n * sizeof(float)));   // выделяем pinned host memory
        std::memcpy(h_pinned, h_in.data(), n * sizeof(float));      // копируем данные в pinned

        // --------- Выделяем память на GPU ---------
        float* d_in = nullptr;                                      // вход на GPU
        float* d_scan_out = nullptr;                                // выход scan на GPU
        CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));           // malloc на GPU для входа
        CUDA_CHECK(cudaMalloc(&d_scan_out, n * sizeof(float)));     // malloc на GPU для выхода scan

        // --------- Копируем данные CPU -> GPU ---------
        CUDA_CHECK(cudaMemcpy(d_in, h_pinned, n * sizeof(float), cudaMemcpyHostToDevice)); // H2D копия

        // --------- GPU reduce: измеряем 3 варианта (global/shared/warp) [Задание 1 + 3] ---------
        float tmp = 0.0f;                                           // временная переменная (для reduce_wrap)

        ReduceArgs a1{ d_in, n, ReduceVariant::GlobalNaive, &tmp };  // аргументы: вариант Global
        float gpu_reduce_global_ms = time_gpu_ms(reduce_wrap, &a1);  // время GPU редукции (global)

        ReduceArgs a2{ d_in, n, ReduceVariant::Shared, &tmp };       // вариант Shared
        float gpu_reduce_shared_ms = time_gpu_ms(reduce_wrap, &a2);  // время GPU редукции (shared)

        ReduceArgs a3{ d_in, n, ReduceVariant::WarpShuffle, &tmp };  // вариант Warp Shuffle
        float gpu_reduce_warp_ms = time_gpu_ms(reduce_wrap, &a3);    // время GPU редукции (warp)

        // Дополнительно получаем сумму (для проверки корректности) — берём лучший вариант WarpShuffle
        float gpu_sum = gpu_reduce_sum(d_in, n, ReduceVariant::WarpShuffle); // сумма GPU
        double rel_err = std::fabs(cpu_sum - (double)gpu_sum) / std::max(1.0, std::fabs(cpu_sum)); // относит. ошибка

        std::cout << std::fixed << std::setprecision(6);            // формат вывода
        std::cout << "CPU sum = " << cpu_sum << "\n";               // сумма CPU
        std::cout << "GPU sum = " << (double)gpu_sum << " (rel_err=" << rel_err << ")\n"; // сумма GPU + ошибка

        // --------- GPU scan: измеряем время [Задание 2 + 3] ---------
        ScanArgs s1{ d_in, d_scan_out, n };                          // аргументы scan
        float gpu_scan_ms = time_gpu_ms(scan_wrap, &s1);             // время GPU scan

        // --------- Копируем результат scan обратно на CPU ---------
        std::vector<float> h_gpu_scan(n);                            // буфер на CPU
        CUDA_CHECK(cudaMemcpy(h_gpu_scan.data(), d_scan_out, n * sizeof(float), cudaMemcpyDeviceToHost)); // D2H

        // --------- Проверяем корректность scan ---------
        bool ok_scan = check_scan(h_cpu_scan, h_gpu_scan);           // сравнение CPU vs GPU
        std::cout << "Scan correctness: " << (ok_scan ? "OK" : "FAIL") << "\n"; // вывод

        // --------- Печатаем времена (для отчёта) ---------
        std::cout << "Time CPU reduce (ms): " << cpu_reduce_ms << "\n";         // CPU reduce
        std::cout << "Time GPU reduce global (ms): " << gpu_reduce_global_ms << "\n"; // GPU global
        std::cout << "Time GPU reduce shared (ms): " << gpu_reduce_shared_ms << "\n"; // GPU shared
        std::cout << "Time GPU reduce warp   (ms): " << gpu_reduce_warp_ms << "\n";   // GPU warp
        std::cout << "Time CPU scan  (ms): " << cpu_scan_ms << "\n";             // CPU scan
        std::cout << "Time GPU scan  (ms): " << gpu_scan_ms << "\n";             // GPU scan

        // --------- Записываем результаты в CSV (потом строим графики) [Задание 3] ---------
        csv << n << ","                                              // размер
            << cpu_reduce_ms << ","                                   // время CPU reduce
            << gpu_reduce_global_ms << ","                            // время GPU reduce (global)
            << gpu_reduce_shared_ms << ","                            // время GPU reduce (shared)
            << gpu_reduce_warp_ms << ","                              // время GPU reduce (warp)
            << cpu_scan_ms << ","                                     // время CPU scan
            << gpu_scan_ms << "\n";                                   // время GPU scan

        // --------- Освобождаем память ---------
        CUDA_CHECK(cudaFree(d_in));                                   // освобождаем GPU вход
        CUDA_CHECK(cudaFree(d_scan_out));                             // освобождаем GPU выход
        CUDA_CHECK(cudaFreeHost(h_pinned));                            // освобождаем pinned host memory
    }

    csv.close();                                                      // закрываем CSV файл
    std::cout << "\nDone. CSV saved: perf.csv\n";                      // сообщение
    return 0;                                                         // успешное завершение
}
