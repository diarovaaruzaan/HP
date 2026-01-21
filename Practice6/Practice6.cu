#define CL_TARGET_OPENCL_VERSION 120                                             // используем OpenCL 1.2
#include <CL/cl.h>                                                               // главный заголовок OpenCL

#include <iostream>                                                              // ввод/вывод
#include <vector>                                                                // std::vector
#include <fstream>                                                               // чтение файлов
#include <string>                                                                // std::string
#include <cmath>                                                                 // fabs
#include <algorithm>                                                             // max

// ---------- макрос проверки ошибок OpenCL ----------
#define CL_CHECK(err, where) do {                                                \
    if ((err) != CL_SUCCESS) {                                                   \
        std::cerr << "OpenCL error " << (err) << " at " << (where) << "\n";      \
        std::exit(1);                                                           \
    }                                                                            \
} while(0)                                                                       // конец макроса

// ---------- читаем kernel.cl как текст ----------
std::string readTextFile(const std::string& path) {                              // функция чтения файла
    std::ifstream f(path);                                                       // открываем файл
    if (!f) {                                                                    // если не открылся
        std::cerr << "Cannot open file: " << path << "\n";                       // выводим ошибку
        std::exit(1);                                                            // выходим
    }
    return std::string((std::istreambuf_iterator<char>(f)),                      // читаем весь файл
                       std::istreambuf_iterator<char>());                        // до конца
}

// ---------- получить имя устройства (CPU/GPU) ----------
std::string getDeviceName(cl_device_id dev) {                                    // функция имени устройства
    size_t sz = 0;                                                               // размер строки
    clGetDeviceInfo(dev, CL_DEVICE_NAME, 0, nullptr, &sz);                       // узнаём размер
    std::string name(sz, '\0');                                                  // создаём строку нужного размера
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sz, name.data(), nullptr);              // читаем имя
    while (!name.empty() && name.back() == '\0') name.pop_back();                // убираем лишние нули
    return name;                                                                 // возвращаем имя
}

// ---------- найти устройство заданного типа (CPU или GPU) ----------
cl_device_id pickDevice(cl_device_type typeWanted) {                             // функция выбора девайса
    cl_uint numPlatforms = 0;                                                    // число платформ
    CL_CHECK(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs");   // получаем количество платформ

    if (numPlatforms == 0) {                                                     // если платформ нет
        return nullptr;                                                          // вернуть nullptr
    }

    std::vector<cl_platform_id> platforms(numPlatforms);                         // массив платформ
    CL_CHECK(clGetPlatformIDs(numPlatforms, platforms.data(), nullptr),          // получаем список платформ
             "clGetPlatformIDs(list)");

    for (auto p : platforms) {                                                   // идём по платформам
        cl_uint numDevices = 0;                                                  // число устройств
        cl_int err = clGetDeviceIDs(p, typeWanted, 0, nullptr, &numDevices);     // пробуем найти устройства

        if (err != CL_SUCCESS || numDevices == 0) {                              // если нет таких устройств
            continue;                                                            // пробуем следующую платформу
        }

        std::vector<cl_device_id> devices(numDevices);                           // массив устройств
        CL_CHECK(clGetDeviceIDs(p, typeWanted, numDevices, devices.data(), nullptr),
                 "clGetDeviceIDs(list)");                                        // берём список устройств
        return devices[0];                                                       // возвращаем первое найденное
    }

    return nullptr;                                                              // если нигде не нашли
}

// ---------- запустить vector_add на выбранном устройстве и вернуть время kernel (мс) ----------
double runVectorAdd(cl_device_id device, const std::string& kernelSource, int n) {   // функция запуска
    cl_int err = CL_SUCCESS;                                                     // переменная для ошибок

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);  // создаём контекст
    CL_CHECK(err, "clCreateContext");                                            // проверяем ошибку

    cl_command_queue queue = clCreateCommandQueue(context, device,               // создаём очередь команд
                                                  CL_QUEUE_PROFILING_ENABLE,    // включаем профилирование
                                                  &err);                        // сюда пишется err
    CL_CHECK(err, "clCreateCommandQueue");                                       // проверяем

    const char* src = kernelSource.c_str();                                      // указатель на текст ядра
    size_t srcLen = kernelSource.size();                                         // длина текста ядра

    cl_program program = clCreateProgramWithSource(context, 1, &src, &srcLen, &err); // создаём программу
    CL_CHECK(err, "clCreateProgramWithSource");                                  // проверяем

    err = clBuildProgram(program, 1, &device, "", nullptr, nullptr);             // компилируем программу под device
    if (err != CL_SUCCESS) {                                                     // если ошибка сборки
        size_t logSize = 0;                                                      // размер лога
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize); // узнаём размер
        std::string log(logSize, '\0');                                          // строка для лога
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr); // лог
        std::cerr << "Build log:\n" << log << "\n";                              // печатаем лог
        CL_CHECK(err, "clBuildProgram");                                         // падаем с ошибкой
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);              // получаем kernel по имени
    CL_CHECK(err, "clCreateKernel(vector_add)");                                 // проверяем

    std::vector<float> A(n), B(n), C(n, 0.0f), Cref(n, 0.0f);                    // создаём массивы на CPU
    for (int i = 0; i < n; i++) {                                                // заполняем данные
        A[i] = (float)i * 0.5f;                                                  // A
        B[i] = (float)i * 0.25f;                                                 // B
        Cref[i] = A[i] + B[i];                                                   // эталон на CPU
    }

    cl_mem dA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, // буфер A на устройстве
                               sizeof(float) * n, (void*)A.data(), &err);
    CL_CHECK(err, "clCreateBuffer(dA)");                                         // проверяем

    cl_mem dB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, // буфер B на устройстве
                               sizeof(float) * n, (void*)B.data(), &err);
    CL_CHECK(err, "clCreateBuffer(dB)");                                         // проверяем

    cl_mem dC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,                       // буфер C на устройстве
                               sizeof(float) * n, nullptr, &err);
    CL_CHECK(err, "clCreateBuffer(dC)");                                         // проверяем

    CL_CHECK(clSetKernelArg(kernel, 0, sizeof(cl_mem), &dA), "clSetKernelArg(0)"); // аргумент 0 = dA
    CL_CHECK(clSetKernelArg(kernel, 1, sizeof(cl_mem), &dB), "clSetKernelArg(1)"); // аргумент 1 = dB
    CL_CHECK(clSetKernelArg(kernel, 2, sizeof(cl_mem), &dC), "clSetKernelArg(2)"); // аргумент 2 = dC
    CL_CHECK(clSetKernelArg(kernel, 3, sizeof(int), &n),    "clSetKernelArg(3)");  // аргумент 3 = n

    size_t local = 256;                                                          // размер локальной группы (work-group)
    size_t global = ( (size_t)n + local - 1 ) / local * local;                   // округляем вверх до кратного local

    cl_event evt;                                                                // событие для профилирования
    CL_CHECK(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr,                   // ставим kernel в очередь
                                    &global, &local, 0, nullptr, &evt),
             "clEnqueueNDRangeKernel");                                          // проверяем

    CL_CHECK(clFinish(queue), "clFinish");                                       // ждём завершения kernel

    CL_CHECK(clEnqueueReadBuffer(queue, dC, CL_TRUE, 0,                          // читаем C обратно на CPU
                                 sizeof(float) * n, C.data(),
                                 0, nullptr, nullptr),
             "clEnqueueReadBuffer");                                             // проверяем

    double maxAbsErr = 0.0;                                                      // максимальная ошибка
    for (int i = 0; i < n; i++) {                                                // сравнение с эталоном
        maxAbsErr = std::max(maxAbsErr, (double)std::fabs(C[i] - Cref[i]));      // обновляем max
    }
    std::cout << "Max abs error: " << maxAbsErr << "\n";                         // печатаем ошибку

    cl_ulong t0 = 0, t1 = 0;                                                     // времена профилирования (нс)
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_START,            // старт
                                     sizeof(cl_ulong), &t0, nullptr),
             "profiling start");                                                 // проверка
    CL_CHECK(clGetEventProfilingInfo(evt, CL_PROFILING_COMMAND_END,              // конец
                                     sizeof(cl_ulong), &t1, nullptr),
             "profiling end");                                                   // проверка

    double ms = (double)(t1 - t0) * 1e-6;                                        // перевод нс -> мс

    clReleaseEvent(evt);                                                         // освобождаем event
    clReleaseMemObject(dA);                                                      // освобождаем буферы
    clReleaseMemObject(dB);                                                      // --
    clReleaseMemObject(dC);                                                      // --
    clReleaseKernel(kernel);                                                     // освобождаем kernel
    clReleaseProgram(program);                                                   // освобождаем program
    clReleaseCommandQueue(queue);                                                // освобождаем queue
    clReleaseContext(context);                                                   // освобождаем context

    return ms;                                                                   // возвращаем время (мс)
}

int main() {                                                                     // точка входа
    const int n = 1 << 20;                                                       // размер массива (1 048 576) для примера
                                                                                // можно увеличить, если нужно
    std::string src = readTextFile("kernel.cl");                                 // читаем kernel.cl

    cl_device_id cpu = pickDevice(CL_DEVICE_TYPE_CPU);                           // пытаемся найти CPU OpenCL
    if (cpu) {                                                                   // если нашли
        std::cout << "CPU device: " << getDeviceName(cpu) << "\n";               // выводим имя
        double msCpu = runVectorAdd(cpu, src, n);                                // запускаем на CPU
        std::cout << "CPU kernel time ms: " << msCpu << "\n\n";                  // печатаем время
    } else {                                                                     // если не нашли CPU
        std::cout << "CPU device not found\n\n";                                 // печать
    }

    cl_device_id gpu = pickDevice(CL_DEVICE_TYPE_GPU);                           // пытаемся найти GPU OpenCL
    if (gpu) {                                                                   // если нашли
        std::cout << "GPU device: " << getDeviceName(gpu) << "\n";               // выводим имя
        double msGpu = runVectorAdd(gpu, src, n);                                // запускаем на GPU
        std::cout << "GPU kernel time ms: " << msGpu << "\n\n";                  // печатаем время
    } else {                                                                     // если не нашли GPU
        std::cout << "GPU device not found\n\n";                                 // печать
    }

    std::cout << "Done\n";                                                       // конец
    return 0;                                                                    // успешное завершение
}
