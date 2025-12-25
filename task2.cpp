#include <iostream>                                     // ввод/вывод
#include <vector>                                       // std::vector
#include <random>                                       // random генератор
#include <chrono>                                       // таймеры
#include <limits>                                       // numeric_limits

int main() {                                            // точка входа

    const int N = 1000000;                              // размер массива
    std::vector<int> arr(N);                            // создаём массив на 1 000 000 элементов

    std::random_device rd;                              // источник энтропии
    std::mt19937 gen(rd());                             // генератор
    std::uniform_int_distribution<int> d(1, 100);        // числа 1..100

    for (int i = 0; i < N; i++) {                       // заполняем массив
        arr[i] = d(gen);                                // пишем случайное число
    }                                                   // конец заполнения

    auto start = std::chrono::high_resolution_clock::now(); // старт времени

    int mn = std::numeric_limits<int>::max();           // начальный минимум
    int mx = std::numeric_limits<int>::min();           // начальный максимум

    for (int i = 0; i < N; i++) {                       // последовательный проход
        if (arr[i] < mn) {                              // проверяем минимум
            mn = arr[i];                                // обновляем минимум
        }                                               // конец if
        if (arr[i] > mx) {                              // проверяем максимум
            mx = arr[i];                                // обновляем максимум
        }                                               // конец if
    }                                                   // конец цикла

    auto end = std::chrono::high_resolution_clock::now(); // конец времени
    std::chrono::duration<double, std::milli> ms = end - start; // время в миллисекундах

    std::cout << "========== TASK 2 ==========" << std::endl;                 // заголовок
    std::cout << "Array size: " << N << " integers" << std::endl;             // вывод размера
    std::cout << "Algorithm: Sequential min/max scan (O(N))" << std::endl;    // пояснение
    std::cout << "Result min: " << mn << std::endl;                           // вывод минимума
    std::cout << "Result max: " << mx << std::endl;                           // вывод максимума
    std::cout << "Time (ms): " << ms.count() << std::endl;                    // вывод времени
    std::cout << "============================" << std::endl;                 // конец блока

    return 0;                                           // завершение
}                                                       // конец main
