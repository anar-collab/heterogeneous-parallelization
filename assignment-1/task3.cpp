#include <iostream>                                     // ввод/вывод
#include <vector>                                       // vector
#include <random>                                       // random
#include <chrono>                                       // таймеры
#include <limits>                                       // limits

#ifdef _OPENMP                                          // если доступен OpenMP
#include <omp.h>                                        // omp функции
#endif                                                  // конец условия

int main() {                                            // точка входа

    const int N = 1000000;                              // размер массива
    std::vector<int> arr(N);                            // массив на N элементов

    std::random_device rd;                              // энтропия
    std::mt19937 gen(rd());                             // генератор
    std::uniform_int_distribution<int> d(1, 100);        // диапазон значений

    for (int i = 0; i < N; i++) {                       // заполняем массив
        arr[i] = d(gen);                                // случайные числа
    }                                                   // конец заполнения

    // ---------------- SEQUENTIAL ----------------
    auto s1 = std::chrono::high_resolution_clock::now(); // старт seq

    int mn_seq = std::numeric_limits<int>::max();       // seq минимум
    int mx_seq = std::numeric_limits<int>::min();       // seq максимум

    for (int i = 0; i < N; i++) {                       // seq проход
        if (arr[i] < mn_seq) {                          // сравнение min
            mn_seq = arr[i];                            // обновление
        }                                               // конец if
        if (arr[i] > mx_seq) {                          // сравнение max
            mx_seq = arr[i];                            // обновление
        }                                               // конец if
    }                                                   // конец цикла

    auto e1 = std::chrono::high_resolution_clock::now(); // конец seq
    std::chrono::duration<double, std::milli> t_seq = e1 - s1; // время seq

    // ---------------- OPENMP ----------------
    auto s2 = std::chrono::high_resolution_clock::now(); // старт omp

    int mn_omp = std::numeric_limits<int>::max();       // omp минимум
    int mx_omp = std::numeric_limits<int>::min();       // omp максимум

#ifdef _OPENMP                                          // если OpenMP доступен
#pragma omp parallel for reduction(min:mn_omp) reduction(max:mx_omp) // параллельный цикл с редукцией
#endif                                                  // конец условия
    for (int i = 0; i < N; i++) {                       // проход по массиву
        if (arr[i] < mn_omp) {                          // сравнение min
            mn_omp = arr[i];                            // обновление (редукция)
        }                                               // конец if
        if (arr[i] > mx_omp) {                          // сравнение max
            mx_omp = arr[i];                            // обновление (редукция)
        }                                               // конец if
    }                                                   // конец цикла

    auto e2 = std::chrono::high_resolution_clock::now(); // конец omp
    std::chrono::duration<double, std::milli> t_omp = e2 - s2; // время omp

    // ---------------- OUTPUT ----------------
    std::cout << "========== TASK 3 ==========" << std::endl;                 // заголовок
    std::cout << "Array size: " << N << " integers" << std::endl;             // размер
    std::cout << "Sequential min/max: min=" << mn_seq << ", max=" << mx_seq << std::endl; // seq результат
    std::cout << "OpenMP min/max:      min=" << mn_omp << ", max=" << mx_omp << std::endl; // omp результат

#ifdef _OPENMP                                          // если OpenMP доступен
    std::cout << "OpenMP threads: " << omp_get_max_threads() << std::endl;    // число потоков
#else                                                   // если OpenMP не включён
    std::cout << "OpenMP threads: OpenMP is OFF (compiled without OpenMP)" << std::endl; // предупреждение
#endif                                                  // конец условия

    std::cout << "Time sequential (ms): " << t_seq.count() << std::endl;      // время seq
    std::cout << "Time OpenMP (ms):      " << t_omp.count() << std::endl;     // время omp

    double speedup = t_seq.count() / t_omp.count();      // считаем ускорение
    std::cout << "Speedup (seq/omp):     " << speedup << "x" << std::endl;    // вывод ускорения
    std::cout << "============================" << std::endl;                 // конец

    return 0;                                            // завершение
}                                                        // конец main
