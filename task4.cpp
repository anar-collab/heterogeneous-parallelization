#include <iostream>                                     // ввод/вывод
#include <vector>                                       // vector
#include <random>                                       // random
#include <chrono>                                       // таймеры

#ifdef _OPENMP                                          // если OpenMP доступен
#include <omp.h>                                        // omp функции
#endif                                                  // конец условия

int main() {                                            // точка входа

    const int N = 5000000;                              // размер массива
    std::vector<int> arr(N);                            // массив на N элементов

    std::random_device rd;                              // энтропия
    std::mt19937 gen(rd());                             // генератор
    std::uniform_int_distribution<int> d(1, 100);        // диапазон значений

    for (int i = 0; i < N; i++) {                       // заполняем массив
        arr[i] = d(gen);                                // случайные числа 1..100
    }                                                   // конец заполнения

    // ---------------- SEQUENTIAL AVERAGE ----------------
    auto s1 = std::chrono::high_resolution_clock::now(); // старт seq

    long long sum_seq = 0;                              // сумма seq
    for (int i = 0; i < N; i++) {                       // seq проход
        sum_seq += arr[i];                              // накапливаем сумму
    }                                                   // конец цикла

    double avg_seq = static_cast<double>(sum_seq) / N;   // среднее seq

    auto e1 = std::chrono::high_resolution_clock::now(); // конец seq
    std::chrono::duration<double, std::milli> t_seq = e1 - s1; // время seq

    // ---------------- OPENMP AVERAGE (REDUCTION) ----------------
    auto s2 = std::chrono::high_resolution_clock::now(); // старт omp

    long long sum_omp = 0;                              // сумма omp

#ifdef _OPENMP                                          // если OpenMP доступен
#pragma omp parallel for reduction(+:sum_omp)            // параллельный цикл с редукцией суммы
#endif                                                  // конец условия
    for (int i = 0; i < N; i++) {                       // проход по массиву
        sum_omp += arr[i];                              // накапливаем сумму (редукция)
    }                                                   // конец цикла

    double avg_omp = static_cast<double>(sum_omp) / N;   // среднее omp

    auto e2 = std::chrono::high_resolution_clock::now(); // конец omp
    std::chrono::duration<double, std::milli> t_omp = e2 - s2; // время omp

    // ---------------- OUTPUT ----------------
    std::cout << "========== TASK 4 ==========" << std::endl;                 // заголовок
    std::cout << "Array size: " << N << " integers" << std::endl;             // размер
    std::cout << "Sequential average: " << avg_seq << std::endl;              // среднее seq

#ifdef _OPENMP                                          // если OpenMP доступен
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;    // потоки
#else                                                   // если OpenMP нет
    std::cout << "OpenMP threads: OpenMP is OFF (compiled without OpenMP)" << std::endl; // предупреждение
#endif                                                  // конец условия

    std::cout << "OpenMP average:      " << avg_omp << std::endl;             // среднее omp
    std::cout << "Time sequential (ms): " << t_seq.count() << std::endl;      // время seq
    std::cout << "Time OpenMP (ms):      " << t_omp.count() << std::endl;     // время omp

    double speedup = t_seq.count() / t_omp.count();      // ускорение
    std::cout << "Speedup (seq/omp):     " << speedup << "x" << std::endl;    // вывод ускорения
    std::cout << "============================" << std::endl;                 // конец

    return 0;                                            // завершение
}                                                        // конец main
