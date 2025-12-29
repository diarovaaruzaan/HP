#include <iostream>     // Библиотека для вывода текста в консоль
#include <chrono>       // Библиотека для измерения времени работы программы
#include <omp.h>        // Библиотека OpenMP для параллельных вычислений

using namespace std;    // Чтобы не писать std:: перед каждой командой

// ---------- ПОСЛЕДОВАТЕЛЬНАЯ СОРТИРОВКА ПУЗЫРЬКОМ ----------
void bubble_simple(int numbers[], int size) {     // Функция сортировки пузырьком
    for (int round = 0; round < size - 1; round++) {   // Количество проходов по массиву
        for (int pos = 0; pos < size - round - 1; pos++) { // Проход по элементам
            if (numbers[pos] > numbers[pos + 1]) {      // Если элементы стоят неправильно
                int temp = numbers[pos];                // Сохраняем текущий элемент
                numbers[pos] = numbers[pos + 1];        // Меняем элементы местами
                numbers[pos + 1] = temp;                // Завершаем обмен
            }
        }
    }
}

// ---------- ПАРАЛЛЕЛЬНАЯ СОРТИРОВКА ПУЗЫРЬКОМ ----------
void bubble_parallel(int numbers[], int size) {     // Параллельная версия пузырька
    for (int round = 0; round < size; round++) {    // Количество проходов
        #pragma omp parallel for                    // Распараллеливаем цикл
        for (int pos = 0; pos < size - 1; pos++) {  // Проход по массиву
            if (numbers[pos] > numbers[pos + 1]) {  // Если элементы не по порядку
                int temp = numbers[pos];            // Сохраняем элемент
                numbers[pos] = numbers[pos + 1];    // Меняем местами
                numbers[pos + 1] = temp;
            }
        }
    }
}

// ---------- ПОСЛЕДОВАТЕЛЬНАЯ СОРТИРОВКА ВЫБОРОМ ----------
void selection_simple(int numbers[], int size) {    // Сортировка выбором
    for (int start = 0; start < size - 1; start++) { // Проходим массив
        int smallest = start;                       // Считаем текущий элемент минимальным
        for (int check = start + 1; check < size; check++) { // Ищем минимум
            if (numbers[check] < numbers[smallest]) {
                smallest = check;                   // Запоминаем индекс минимума
            }
        }
        int temp = numbers[start];                  // Меняем элементы местами
        numbers[start] = numbers[smallest];
        numbers[smallest] = temp;
    }
}

// ---------- ПАРАЛЛЕЛЬНАЯ СОРТИРОВКА ВЫБОРОМ ----------
void selection_parallel(int numbers[], int size) {  // Параллельная версия
    for (int start = 0; start < size - 1; start++) {
        int smallest = start;                       // Минимальный элемент

        #pragma omp parallel for                    // Параллельный поиск минимума
        for (int check = start + 1; check < size; check++) {
            if (numbers[check] < numbers[smallest]) {
                #pragma omp critical                // Запрещаем одновременный доступ
                smallest = check;
            }
        }

        int temp = numbers[start];                  // Обмен элементов
        numbers[start] = numbers[smallest];
        numbers[smallest] = temp;
    }
}

// ---------- ПОСЛЕДОВАТЕЛЬНАЯ СОРТИРОВКА ВСТАВКАМИ ----------
void insertion_simple(int numbers[], int size) {    // Сортировка вставками
    for (int i = 1; i < size; i++) {                // Начинаем со второго элемента
        int current = numbers[i];                   // Текущий элемент
        int j = i - 1;                              // Индекс слева

        while (j >= 0 && numbers[j] > current) {    // Пока элементы больше
            numbers[j + 1] = numbers[j];            // Сдвигаем вправо
            j--;
        }
        numbers[j + 1] = current;                   // Вставляем элемент
    }
}

// ---------- ИЗМЕРЕНИЕ ВРЕМЕНИ ----------
double check_time(void (*sort_func)(int[], int), int data[], int size) {
    auto begin = chrono::high_resolution_clock::now(); // Запоминаем начало
    sort_func(data, size);                              // Запускаем сортировку
    auto finish = chrono::high_resolution_clock::now(); // Конец
    return chrono::duration<double>(finish - begin).count(); // Считаем время
}

// ---------- MAIN ----------
int main() {
    int test_sizes[3] = {1000, 10000, 100000};      // Размеры массивов

    for (int t = 0; t < 3; t++) {                   // Проходим по размерам
        int size = test_sizes[t];                   // Текущий размер
        int* base_array = new int[size];            // Создаем массив

        for (int i = 0; i < size; i++) {            // Заполняем массив
            base_array[i] = rand() % 100;            // Случайные числа
        }

        cout << "\nРазмер массива: " << size << endl;

        int* temp_array = new int[size];            // Массив для сортировки
        for (int i = 0; i < size; i++) temp_array[i] = base_array[i];
        cout << "Bubble simple: "
             << check_time(bubble_simple, temp_array, size) << " сек\n";

        for (int i = 0; i < size; i++) temp_array[i] = base_array[i];
        cout << "Bubble parallel: "
             << check_time(bubble_parallel, temp_array, size) << " сек\n";

        for (int i = 0; i < size; i++) temp_array[i] = base_array[i];
        cout << "Selection simple: "
             << check_time(selection_simple, temp_array, size) << " сек\n";

        for (int i = 0; i < size; i++) temp_array[i] = base_array[i];
        cout << "Selection parallel: "
             << check_time(selection_parallel, temp_array, size) << " сек\n";

        for (int i = 0; i < size; i++) temp_array[i] = base_array[i];
        cout << "Insertion simple: "
             << check_time(insertion_simple, temp_array, size) << " сек\n";

        delete[] base_array;                         // Освобождаем память
        delete[] temp_array;
    }

    return 0;                                       // Завершаем программу
}
