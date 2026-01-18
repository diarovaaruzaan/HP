#include <iostream>                   // Подключаем библиотеку для ввода/вывода (cout)
#include <cuda_runtime.h>             // Подключаем библиотеки CUDA для работы с GPU
#include <chrono>                     // Для замеров времени на CPU
using namespace std;                  // Чтобы не писать std:: перед cout, endl и т.д.
using namespace std::chrono;          // Чтобы не писать std::chrono:: перед таймерами

// Макрос для проверки ошибок CUDA
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); } // Проверка результата CUDA вызова
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if(code != cudaSuccess){                                  // Если произошла ошибка
        fprintf(stderr,"GPUassert: %s %s %d\n",              // Выводим текст ошибки, файл и строку
                cudaGetErrorString(code), file, line);
        if(abort) exit(code);                                 // Если abort=true, завершаем программу
    }
}

// ================= Task 1: умножение массива =================
__global__ void multiply_kernel(float* data, float value, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;         // Вычисляем глобальный индекс потока
    if(idx < n) data[idx] *= value;                          // Если индекс в пределах массива, умножаем элемент
}

// ================= Task 2: сложение массивов =================
__global__ void add_kernel(float* a, float* b, float* c, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;         // Вычисляем глобальный индекс потока
    if(idx < n) c[idx] = a[idx] + b[idx];                   // Складываем элементы массивов
}

// ================= Task 3: коалесцированный доступ =================
__global__ void coalesced_kernel(float* data, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;         // Вычисляем индекс
    if(idx < n) data[idx] *= 2.0f;                           // Умножаем элемент на 2
}

// ================= Task 3: некоалесцированный доступ =================
__global__ void non_coalesced_kernel(float* data, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;         // Вычисляем индекс потока
    int newIdx = (idx * 32) % n;                             // Нарушаем последовательность для имитации некоалесцированного доступа
    if(idx < n) data[newIdx] *= 2.0f;                        // Модифицируем элемент с «разбросанным» индексом
}

// ================= MAIN =================
int main(){
    const int N = 1000000;                                   // Размер массива 1 миллион элементов
    size_t size = N * sizeof(float);                         // Размер массива в байтах

    // Проверка наличия GPU
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);                        // Получаем количество GPU в системе
    bool useGPU = deviceCount > 0;                           // Если есть хотя бы один GPU, используем CUDA
    if(!useGPU){
        cout << "GPU не найден, выполняем на CPU!\n";        // Сообщение, если GPU нет
    } else {
        cout << "GPU найден, выполняем на CUDA!\n";          // Сообщение о наличии GPU
    }

    // ================= Создаём хост-массивы =================
    float *h_data = new float[N];                             // Массив для Task 1 и Task 3
    float *h_a = new float[N];                                // Массив a для Task 2 и Task 4
    float *h_b = new float[N];                                // Массив b для Task 2 и Task 4
    float *h_c = new float[N];                                // Массив c для результатов сложения

    // Инициализация массивов
    for(int i=0;i<N;i++){
        h_data[i] = 1.0f;                                     // Заполняем массив единицами
        h_a[i] = 1.0f;                                        // Массив a = 1
        h_b[i] = 2.0f;                                        // Массив b = 2
    }

    if(useGPU){                                                 // Если GPU доступен
        // ================= Выделение GPU памяти =================
        float *d_data, *d_a, *d_b, *d_c;                       
        CUDA_CHECK(cudaMalloc(&d_data,size));                 // Выделяем память на GPU для Task 1/3
        CUDA_CHECK(cudaMalloc(&d_a,size));                    // Выделяем память на GPU для Task 2/4
        CUDA_CHECK(cudaMalloc(&d_b,size));                    // Выделяем память на GPU для Task 2/4
        CUDA_CHECK(cudaMalloc(&d_c,size));                    // Выделяем память на GPU для результатов сложения

        // Копируем массивы с CPU на GPU
        CUDA_CHECK(cudaMemcpy(d_data,h_data,size,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_a,h_a,size,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b,h_b,size,cudaMemcpyHostToDevice));

        // Таймеры CUDA
        cudaEvent_t start, stop;                              // События CUDA для измерения времени
        CUDA_CHECK(cudaEventCreate(&start));                 
        CUDA_CHECK(cudaEventCreate(&stop));

        int threads = 256;                                    // Потоки на блок
        int blocks = (N + threads - 1) / threads;            // Количество блоков, чтобы покрыть все элементы

        // ================= Task 1 =================
        CUDA_CHECK(cudaEventRecord(start));                   
        multiply_kernel<<<blocks,threads>>>(d_data,2.0f,N);  // Запускаем ядро на GPU
        CUDA_CHECK(cudaDeviceSynchronize());                 // Ждем завершения всех потоков
        CUDA_CHECK(cudaEventRecord(stop));                   
        CUDA_CHECK(cudaEventSynchronize(stop));             
        float time1;                                        
        CUDA_CHECK(cudaEventElapsedTime(&time1,start,stop)); // Считаем время выполнения в миллисекундах

        // Второй запуск для сравнения
        CUDA_CHECK(cudaEventRecord(start));                   
        multiply_kernel<<<blocks,threads>>>(d_data,2.0f,N);  
        CUDA_CHECK(cudaDeviceSynchronize());                 
        CUDA_CHECK(cudaEventRecord(stop));                   
        CUDA_CHECK(cudaEventSynchronize(stop));             
        float time1_sim;
        CUDA_CHECK(cudaEventElapsedTime(&time1_sim,start,stop));

        cout << "Task 1:\n Global multiply: " << time1 << " ms\n Second run: " << time1_sim << " ms\n\n";

        // ================= Task 2 =================
        int blockSizes[3] = {128,256,512};                        
        cout << "Task 2 (different block sizes):\n";
        for(int i=0;i<3;i++){
            threads = blockSizes[i];                            // Меняем размер блока
            blocks = (N + threads - 1) / threads;              // Считаем количество блоков
            CUDA_CHECK(cudaEventRecord(start));
            add_kernel<<<blocks,threads>>>(d_a,d_b,d_c,N);    // Складываем массивы
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            float time;
            CUDA_CHECK(cudaEventElapsedTime(&time,start,stop));
            cout << " Block size " << threads << ": " << time << " ms\n"; // Вывод времени
        }
        cout << endl;

        // ================= Task 3 =================
        CUDA_CHECK(cudaEventRecord(start));
        coalesced_kernel<<<blocks,256>>>(d_data,N);           // Коалесцированный доступ
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float tCoalesced;
        CUDA_CHECK(cudaEventElapsedTime(&tCoalesced,start,stop));

        CUDA_CHECK(cudaEventRecord(start));
        non_coalesced_kernel<<<blocks,256>>>(d_data,N);       // Некоалесцированный доступ
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float tNonCoalesced;
        CUDA_CHECK(cudaEventElapsedTime(&tNonCoalesced,start,stop));

        cout << "Task 3:\n Coalesced: " << tCoalesced << " ms\n Non-coalesced: " << tNonCoalesced << " ms\n\n";

        // ================= Task 4 =================
        CUDA_CHECK(cudaEventRecord(start));
        add_kernel<<<blocks,64>>>(d_a,d_b,d_c,N);             // Неоптимальная конфигурация блоков
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float tBad;
        CUDA_CHECK(cudaEventElapsedTime(&tBad,start,stop));

        CUDA_CHECK(cudaEventRecord(start));
        add_kernel<<<blocks,256>>>(d_a,d_b,d_c,N);            // Оптимальная конфигурация блоков
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float tGood;
        CUDA_CHECK(cudaEventElapsedTime(&tGood,start,stop));

        cout << "Task 4:\n Non-optimal: " << tBad << " ms\n Optimal: " << tGood << " ms\n\n";

        // ================= Очистка GPU памяти =================
        cudaFree(d_data); cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    } else {
        // ================= CPU fallback =================
        cout << "CPU fallback (дробные миллисекунды)\n";

        // ================= Task 1 =================
        auto cpuStart = high_resolution_clock::now();          // Начало таймера
        for(int i=0;i<N;i++) h_data[i]*=2.0f;                  // Умножаем массив
        auto cpuStop = high_resolution_clock::now();           // Конец таймера
        double time1 = duration<double,milli>(cpuStop-cpuStart).count(); // Считаем время в ms
        cout << "Task 1 CPU multiply - run 1: " << time1 << " ms\n";

        cpuStart = high_resolution_clock::now();
        for(int i=0;i<N;i++) h_data[i]*=2.0f;                 
        cpuStop = high_resolution_clock::now();
        double time1_sim = duration<double,milli>(cpuStop-cpuStart).count();
        cout << "Task 1 CPU multiply - run 2: " << time1_sim << " ms\n";

        // ================= Task 2 =================
        int blockSizes[3] = {128,256,512};
        for(int i=0;i<3;i++){
            cpuStart = high_resolution_clock::now();
            for(int j=0;j<N;j++) h_c[j] = h_a[j] + h_b[j];     // Складываем массивы
            cpuStop = high_resolution_clock::now();
            double t = duration<double,milli>(cpuStop-cpuStart).count();
            cout << "Task 2 CPU add - block size " << blockSizes[i] << ": " << t << " ms\n";
        }

        // ================= Task 3 =================
        cpuStart = high_resolution_clock::now();
        for(int i=0;i<N;i++) h_data[i] *= 2.0f;                 // Коалесцированный доступ
        cpuStop = high_resolution_clock::now();
        cout << "Task 3 CPU coalesced: " 
             << duration<double,milli>(cpuStop-cpuStart).count() << " ms\n";

        cpuStart = high_resolution_clock::now();
        for(int i=0;i<N;i++) h_data[(i*32)%N] *= 2.0f;         // Некоалесцированный доступ
        cpuStop = high_resolution_clock::now();
        cout << "Task 3 CPU non-coalesced: " 
             << duration<double,milli>(cpuStop-cpuStart).count() << " ms\n";

        // ================= Task 4 =================
        cpuStart = high_resolution_clock::now();
        for(int i=0;i<N;i++) h_c[i] = h_a[i] + h_b[i];         // Неоптимальная конфигурация
        cpuStop = high_resolution_clock::now();
        cout << "Task 4 CPU non-optimal: " 
             << duration<double,milli>(cpuStop-cpuStart).count() << " ms\n";

        cpuStart = high_resolution_clock::now();
        for(int i=0;i<N;i++) h_c[i] = h_a[i] + h_b[i];         // Оптимальная конфигурация
        cpuStop = high_resolution_clock::now();
        cout << "Task 4 CPU optimal: " 
             << duration<double,milli>(cpuStop-cpuStart).count() << " ms\n";
    }

    // ================= Очистка хост-массивов =================
    delete[] h_data; delete[] h_a; delete[] h_b; delete[] h_c;   // Освобождаем память на CPU

    return 0;                                                     // Завершаем программу
}
