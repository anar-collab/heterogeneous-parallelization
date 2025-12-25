#include <iostream>                         // подключаем ввод/вывод
#include <random>                           // подключаем генератор случайных чисел

int main() {                                // точка входа

    const int N = 50000;                    // размер массива по заданию
    int* arr = new int[N];                  // динамически выделяем память под N целых чисел

    std::random_device rd;                  // источник энтропии
    std::mt19937 gen(rd());                 // генератор Mersenne Twister
    std::uniform_int_distribution<int> d(1, 100); // распределение от 1 до 100

    long long sum = 0;                      // сумма элементов (long long чтобы не было переполнения)

    for (int i = 0; i < N; i++) {           // цикл по всем элементам массива
        arr[i] = d(gen);                    // заполняем случайным числом 1..100
        sum += arr[i];                      // добавляем в сумму
    }                                       // конец цикла

    double avg = static_cast<double>(sum) / N; // вычисляем среднее значение

    std::cout << "========== TASK 1 ==========" << std::endl;                 // заголовок
    std::cout << "Array size: " << N << " integers" << std::endl;             // вывод размера
    std::cout << "Sum: " << sum << std::endl;                                 // вывод суммы
    std::cout << "Average: " << avg << std::endl;                             // вывод среднего
    std::cout << "============================" << std::endl;                 // конец блока

    delete[] arr;                           // освобождаем память
    arr = nullptr;                          // зануляем указатель

    return 0;                               // успешное завершение
}                                           // конец main
