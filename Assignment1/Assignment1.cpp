#include <iostream>   
#include <vector>     
#include <random>    
#include <chrono>     
#ifdef _OPENMP
#include <omp.h>      
#endif

int main() {
    using namespace std; // Чтобы не писать std:: перед cout, vector и другими штуками
    // --- Настройки диапазона случайных чисел ---
    constexpr int RAND_MIN_VAL = 1;      // Минимум случайного числа
    constexpr int RAND_MAX_VAL = 100;    // Максимум случайного числа
    // Порог вывода массива: если N <= PRINT_LIMIT, массив печатается на экран
    constexpr size_t PRINT_LIMIT = 100;  // <-- можно менять, если хотите печатать большие массивы

    
    // ---------------------------
    // Задание 1: среднее из 50 000 чисел
    // ---------------------------


    size_t N1 = 50000;                   // Размер списка 50 тысяч
    vector<int> arr1(N1);                // Создаём пустой список на 50 тысяч чисел
    random_device rd;                     // Берём случайную “зернышко” для генератора
    mt19937 gen(rd());                    // Создаём генератор случайных чисел
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL); // Скажем генератору что числа от 1 до 100

    for (size_t i = 0; i < N1; ++i)      // Проходим по всем 50 тысячам чисел
        arr1[i] = dist(gen);             // И кладём туда случайное число
    if (N1 <= PRINT_LIMIT) {             // Если список маленький, то показываем его
        cout << "Array 1: ";             // Пишем “Array 1:”
        for (auto v : arr1) cout << v << ' '; // Показываем все числа через пробел
        cout << '\n';                     // Переходим на новую строчку
    }

    double sum1 = 0;                      // Заводим переменную для суммы чисел
    auto t1_start = chrono::high_resolution_clock::now(); // Запоминаем время начала подсчёта
    for (size_t i = 0; i < N1; ++i)       // Проходим по всем числам
        sum1 += arr1[i];                  // Прибавляем число к сумме
    double mean1 = sum1 / N1;             // Делим сумму на количество чисел, получаем среднее
    auto t1_end = chrono::high_resolution_clock::now();   // Запоминаем время окончания
    chrono::duration<double, milli> dur1 = t1_end - t1_start; // Считаем сколько миллисекунд прошло

    cout << "Task 1: mean = " << mean1 << ", time = " << dur1.count() << " ms\n"; // Показываем среднее и время


    // ---------------------------
    // Задание 2: последовательный поиск минимального и максимального числа
    // ---------------------------


    size_t N2 = 1000000;                  // Размер списка 1 миллион
    vector<int> arr2(N2);                 // Создаём пустой список на миллион чисел
    for (size_t i = 0; i < N2; ++i)       // Проходим по каждому месту
        arr2[i] = dist(gen);              // Кладём туда случайное число

    int min_seq = arr2[0];                 // Первое число и считаем его самым маленьким
    int max_seq = arr2[0];                 // первое число и считаем его самым большим
    auto t2_start = chrono::high_resolution_clock::now(); // время начала поиска
    for (size_t i = 1; i < N2; ++i) {     // проверяем все отсальные числа
        if (arr2[i] < min_seq) min_seq = arr2[i]; // Число меньше, заменяем min
        if (arr2[i] > max_seq) max_seq = arr2[i]; // Число больше, заменяем max
    }
    auto t2_end = chrono::high_resolution_clock::now(); // Время конца поиска
    chrono::duration<double, milli> dur2 = t2_end - t2_start; // считаем сколько времени прошло

    cout << "Task 2 Sequential: min = " << min_seq << ", max = " << max_seq
         << ", time = " << dur2.count() << " ms\n";       // Результаты


    // ---------------------------
    // Задание 3: параллельный поиск min/max с OpenMP
    // ---------------------------


    int min_par = arr2[0];                 // Первое число и считаем его самым маленьким
    int max_par = arr2[0];                 // первое число и считаем его самым большим
    auto t3_start = chrono::high_resolution_clock::now(); // время начала поиска

    #ifdef _OPENMP // Параллельный цикл с OpenMP, каждый поток считает свой min и max, потом объединяются
    #pragma omp parallel for reduction(min: min_par) reduction(max: max_par)
    for (int i = 1; i < static_cast<int>(N2); ++i) { // Ппроверяем все отсальные числа
        if (arr2[i] < min_par) min_par = arr2[i];    // Число меньше, заменяем min
        if (arr2[i] > max_par) max_par = arr2[i];    // ЕЧисло меньше, заменяем max
    #else
    // Если OpenMP нет, делаем то же самое обычным циклом
    for (size_t i = 1; i < N2; ++i) {
        if (arr2[i] < min_par) min_par = arr2[i];
        if (arr2[i] > max_par) max_par = arr2[i];
    }
    #endif

    auto t3_end = chrono::high_resolution_clock::now(); // Запоминаем время конца
    chrono::duration<double, milli> dur3 = t3_end - t3_start; // Считаем сколько времени 

    cout << "Task 3 Parallel:   min = " << min_par << ", max = " << max_par
        << ", time = " << dur3.count() << " ms\n";       // результаты

    // ---------------------------
    // Задание 4: среднее значение массива из 5 000 000 чисел
    // ---------------------------
    
    size_t N3 = 5000000;                  // Размер массива
    vector<int> arr3(N3);                 // Создаём массив на 5 миллионов чисел
    for (size_t i = 0; i < N3; ++i)       // Заполняем числами от 1 до 100
        arr3[i] = dist(gen);

    double sum_seq = 0;                    // Переменная для последовательной суммы
    auto t4_start = chrono::high_resolution_clock::now(); // Время начала
    for (size_t i = 0; i < N3; ++i)        // Идём по всем числам
        sum_seq += arr3[i];               // Складываем их
    double mean_seq = sum_seq / N3;       // Делим на количество чисел, получаем среднее
    auto t4_end = chrono::high_resolution_clock::now();   // Время конца
    chrono::duration<double, milli> dur4 = t4_end - t4_start; // Считаем время

    double sum_par = 0;                    // Переменная для параллельной суммы
    auto t5_start = chrono::high_resolution_clock::now(); // Время начала
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+: sum_par) // Скажем OpenMP, что сумма будет считаться параллельно
    for (int i = 0; i < static_cast<int>(N3); ++i) // Каждый поток идёт по своей части массива
        sum_par += arr3[i];                // Складываем числа
    #else
    for (size_t i = 0; i < N3; ++i)       // Если OpenMP нет, делаем обычным циклом
        sum_par += arr3[i];
    #endif
    double mean_par = sum_par / N3;       // Делим на количество чисел, получаем среднее
    auto t5_end = chrono::high_resolution_clock::now();   // Время конца
    chrono::duration<double, milli> dur5 = t5_end - t5_start; // Считаем время

    cout << "Task 4 Sequential mean = " << mean_seq << ", time = " << dur4.count() << " ms\n"; // Показываем последовательное
    cout << "Task 4 Parallel mean   = " << mean_par << ", time = " << dur5.count() << " ms\n"; // Показываем параллельное

    #ifdef _OPENMP
    cout << "OpenMP threads (max): " << omp_get_max_threads() << '\n'; // Показываем, сколько потоков OpenMP использует
    #endif

    return 0;                              // Заканчиваем программу

}
