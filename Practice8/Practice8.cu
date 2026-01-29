// ======================= Практическая работа №8 =======================
// Гибридное приложение CPU(OpenMP) + GPU(CUDA)
// Задание 1: CPU обработка с OpenMP
// Задание 2: GPU обработка с CUDA
// Задание 3: Гибрид CPU+GPU одновременно + передача данных
// Задание 4: Анализ производительности + сравнение времени + проверка
// =====================================================================

#include <cuda_runtime.h>              // CUDA runtime API (cudaMalloc, cudaMemcpy, cudaEvent...)
#include <device_launch_parameters.h>  // threadIdx, blockIdx, blockDim, gridDim

#include <iostream>   // вывод в консоль
#include <vector>     // динамические массивы std::vector
#include <random>     // генерация случайных чисел
#include <chrono>     // замеры времени на CPU
#include <iomanip>    // setprecision для красивого вывода
#include <cmath>      // fabs, max

#ifdef _OPENMP
#include <omp.h>      // OpenMP (omp_get_max_threads, pragma omp parallel for)
#endif

// -------------------- Макрос проверки ошибок CUDA --------------------
// Любой вызов CUDA оборачиваем в CUDA_CHECK(...)
// Если ошибка есть -> выводим текст ошибки и завершаем программу.
#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

// =====================================================================
// ===================== ЗАДАНИЕ 2: CUDA kernel =========================
// =====================================================================

// GPU-ядро: умножить каждый элемент на 2
// in  - входной массив на GPU
// out - выходной массив на GPU
// n   - размер массива
__global__ void mul2_kernel(const float* __restrict__ in,
                            float* __restrict__ out,
                            int n)
{
    // Глобальный индекс потока (какой элемент он обрабатывает)
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Защита от выхода за границы массива
    if (i < n) out[i] = in[i] * 2.0f;
}

// =====================================================================
// ===================== ЗАДАНИЕ 1: CPU OpenMP ==========================
// =====================================================================

// CPU-функция обработки массива: умножение на 2
// OpenMP распараллеливает цикл по i
void cpu_mul2_omp(const float* in, float* out, int n)
{
    // Каждый поток CPU обрабатывает часть массива
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        out[i] = in[i] * 2.0f;
    }
}

// =====================================================================
// ======================= Таймер для CPU ===============================
// =====================================================================

// Универсальная функция для измерения времени CPU участка кода
template <class F>
double time_cpu_ms(F&& fn)
{
    auto t0 = std::chrono::high_resolution_clock::now(); // старт
    fn();                                                 // выполняем код
    auto t1 = std::chrono::high_resolution_clock::now(); // конец
    std::chrono::duration<double, std::milli> dt = t1 - t0;
    return dt.count();                                    // возвращаем миллисекунды
}

// =====================================================================
// ======================= Таймер для GPU ===============================
// =====================================================================

// Таймер на GPU через cudaEvent: измеряет время выполнения kernel/участка на GPU
struct GpuTimer {
    cudaEvent_t start{}, stop{};   // события CUDA

    // создаём события
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }

    // удаляем события (чтобы не было утечек)
    ~GpuTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // начинаем отсчёт
    void begin(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(start, s)); }

    // заканчиваем отсчёт и возвращаем миллисекунды
    float end(cudaStream_t s = 0) {
        CUDA_CHECK(cudaEventRecord(stop, s));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms;
    }
};

// =====================================================================
// ===================== Проверка корректности ==========================
// =====================================================================

// Сравнение двух массивов (CPU эталон vs GPU/Hybrid результат)
// eps - допустимая относительная погрешность (float может немного отличаться)
bool check_equal(const std::vector<float>& a, const std::vector<float>& b, float eps = 1e-5f)
{
    if (a.size() != b.size()) return false;

    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::fabs(a[i] - b[i]);
        float norm = std::max(1.0f, std::max(std::fabs(a[i]), std::fabs(b[i])));

        // Если относительная ошибка больше eps -> ошибка
        if (diff / norm > eps) {
            std::cerr << "Mismatch at i=" << i << " a=" << a[i] << " b=" << b[i] << "\n";
            return false;
        }
    }
    return true;
}

// =====================================================================
// =============================== MAIN =================================
// =====================================================================

int main()
{
    // Размер массива (можно менять для экспериментов)
    const int N = 1'000'000;

    // Вывод информации по OpenMP
#ifdef _OPENMP
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP: OFF (compiled without -fopenmp)\n";
#endif

    std::cout << "N = " << N << "\n";

    // -------------------- Создаём входные данные на CPU --------------------
    std::vector<float> h_in(N);                       // входной массив на CPU
    std::mt19937 rng(123);                            // генератор (seed=123)
    std::uniform_real_distribution<float> dist(0.0f, 1.0f); // числа 0..1
    for (int i = 0; i < N; ++i) h_in[i] = dist(rng);  // заполняем случайными значениями

    // Выходы (CPU / GPU / Hybrid)
    std::vector<float> h_out_cpu(N);     // результат CPU
    std::vector<float> h_out_gpu(N);     // результат GPU
    std::vector<float> h_out_hybrid(N);  // результат Hybrid

    // =================================================================
    // ===================== ЗАДАНИЕ 1: CPU + OpenMP ====================
    // =================================================================
    double cpu_ms = time_cpu_ms([&]() {
        cpu_mul2_omp(h_in.data(), h_out_cpu.data(), N); // CPU обработка
    });

    // =================================================================
    // ===================== ЗАДАНИЕ 2: GPU + CUDA ======================
    // =================================================================

    // Указатели на память GPU
    float *d_in = nullptr, *d_out = nullptr;

    // Выделяем память на GPU
    CUDA_CHECK(cudaMalloc(&d_in,  N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));

    // Копируем входные данные CPU -> GPU (Host to Device)
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), N * sizeof(float), cudaMemcpyHostToDevice));

    // Настройки запуска kernel
    int threads = 256;                      // потоков в блоке
    int blocks  = (N + threads - 1) / threads; // число блоков

    // Замер времени выполнения kernel на GPU (ТОЛЬКО KERNEL)
    GpuTimer gt_gpu;
    gt_gpu.begin();
    mul2_kernel<<<blocks, threads>>>(d_in, d_out, N); // запуск kernel
    CUDA_CHECK(cudaGetLastError());                   // проверка ошибок запуска
    CUDA_CHECK(cudaDeviceSynchronize());              // ждём завершения kernel (для честного замера)
    float gpu_kernel_ms = gt_gpu.end();               // время kernel

    // Копируем результат GPU -> CPU (Device to Host)
    CUDA_CHECK(cudaMemcpy(h_out_gpu.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    // =================================================================
    // ================= ЗАДАНИЕ 3: ГИБРИД CPU + GPU ====================
    // =================================================================
    // Первая половина массива -> CPU (OpenMP)
    // Вторая половина массива -> GPU (CUDA) с pinned memory + stream

    const int N_cpu = N / 2;       // сколько элементов обрабатывает CPU
    const int N_gpu = N - N_cpu;   // сколько элементов обрабатывает GPU

    // Pinned memory (закреплённая память) ускоряет копирование CPU<->GPU
    float* h_pinned_in  = nullptr; // pinned вход (вторая половина)
    float* h_pinned_out = nullptr; // pinned выход (вторая половина)
    CUDA_CHECK(cudaMallocHost(&h_pinned_in,  N_gpu * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_pinned_out, N_gpu * sizeof(float)));

    // Копируем вторую половину входного массива в pinned буфер
    memcpy(h_pinned_in, h_in.data() + N_cpu, N_gpu * sizeof(float));

    // Создаём CUDA stream, чтобы делать асинхронные копирования и kernel
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Память на GPU для второй половины
    float *d_in2 = nullptr, *d_out2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in2,  N_gpu * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out2, N_gpu * sizeof(float)));

    // Общий таймер гибридного режима (CPU + GPU вместе)
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---------- GPU часть (асинхронно в stream) ----------
    // H2D копирование pinned -> GPU
    CUDA_CHECK(cudaMemcpyAsync(d_in2, h_pinned_in, N_gpu * sizeof(float),
                               cudaMemcpyHostToDevice, stream));

    // запуск kernel для второй половины
    int blocks2 = (N_gpu + threads - 1) / threads;
    mul2_kernel<<<blocks2, threads, 0, stream>>>(d_in2, d_out2, N_gpu);
    CUDA_CHECK(cudaGetLastError());

    // D2H копирование результата GPU -> pinned
    CUDA_CHECK(cudaMemcpyAsync(h_pinned_out, d_out2, N_gpu * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));

    // ---------- CPU часть (в это же время CPU считает свою половину) ----------
    cpu_mul2_omp(h_in.data(), h_out_hybrid.data(), N_cpu);

    // Ждём завершения всех операций в stream (копии + kernel)
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Переносим вторую половину результата из pinned в общий выход
    memcpy(h_out_hybrid.data() + N_cpu, h_pinned_out, N_gpu * sizeof(float));

    // Останавливаем таймер гибридного режима
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> hybrid_dt = t1 - t0;
    double hybrid_ms = hybrid_dt.count();

    // =================================================================
    // ============= ЗАДАНИЕ 4: Анализ + проверка корректности ==========
    // =================================================================

    // Проверяем результаты GPU и Hybrid относительно CPU (эталон)
    bool ok_gpu    = check_equal(h_out_cpu, h_out_gpu);
    bool ok_hybrid = check_equal(h_out_cpu, h_out_hybrid);

    // Красивый вывод
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nCorrectness GPU:    " << (ok_gpu ? "OK" : "FAIL") << "\n";
    std::cout << "Correctness Hybrid: " << (ok_hybrid ? "OK" : "FAIL") << "\n\n";

    // Печатаем времена выполнения
    std::cout << "Time CPU OpenMP (ms):      " << cpu_ms << "\n";
    std::cout << "Time GPU kernel only (ms): " << gpu_kernel_ms << "\n";
    std::cout << "Time Hybrid total (ms):    " << hybrid_ms << "\n";

    // Ускорение гибрид относительно CPU (если меньше 1 -> гибрид медленнее)
    std::cout << "\nSpeedup Hybrid vs CPU: " << (cpu_ms / hybrid_ms) << "x\n";

    // -------------------- Освобождение ресурсов --------------------
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_in2));
    CUDA_CHECK(cudaFree(d_out2));

    CUDA_CHECK(cudaFreeHost(h_pinned_in));
    CUDA_CHECK(cudaFreeHost(h_pinned_out));

    CUDA_CHECK(cudaStreamDestroy(stream));

    return 0; // завершение программы
}
