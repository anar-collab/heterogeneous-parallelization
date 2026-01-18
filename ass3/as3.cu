// =============================== 
// Assignment 3: GPU Architecture & CUDA Optimization
// Full solution for Tasks 1–4
// Each line is commented (no “комментарий” word)
// ===============================

#include <cuda_runtime.h>                 // CUDA Runtime API
#include <device_launch_parameters.h>     // Kernel launch params (MSVC)
#include <iostream>                       // std::cout
#include <vector>                         // std::vector
#include <iomanip>                        // std::fixed, std::setprecision
#include <algorithm>                      // std::min, std::max
#include <cstdlib>                        // std::exit

#define CUDA_CHECK(call) do {                                                /* CUDA error check wrapper */ \
    cudaError_t err = (call);                                                /* execute CUDA call */ \
    if (err != cudaSuccess) {                                                /* if failed */ \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; /* print reason */ \
        std::exit(EXIT_FAILURE);                                             /* stop program */ \
    }                                                                        /* end if */ \
} while (0)                                                                  /* end macro */

static const int N = 1'000'000;              // array size
static const float SCALE_K = 1.2345f;        // scale factor for Task 1

__global__ void scale_global(float* data, float k, int n) {                   // Task 1: global-only scaling
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          // global thread index
    if (idx < n) {                                                           // bounds check
        data[idx] = data[idx] * k;                                           // elementwise multiply (global mem)
    }                                                                        // end bounds
}                                                                            // end kernel

__global__ void scale_shared(float* data, float k, int n) {                   // Task 1: shared-memory scaling
    extern __shared__ float tile[];                                          // dynamic shared memory tile
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          // global thread index
    int tid = threadIdx.x;                                                   // local thread index
    if (idx < n) {                                                           // bounds check
        tile[tid] = data[idx];                                               // load from global to shared
    }                                                                        // end bounds
    __syncthreads();                                                         // sync threads in block
    if (idx < n) {                                                           // bounds check
        tile[tid] = tile[tid] * k;                                           // compute in shared
    }                                                                        // end bounds
    __syncthreads();                                                         // sync before store
    if (idx < n) {                                                           // bounds check
        data[idx] = tile[tid];                                               // store back to global
    }                                                                        // end bounds
}                                                                            // end kernel

__global__ void vec_add(const float* a, const float* b, float* c, int n) {     // Task 2/4: vector addition
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          // global thread index
    if (idx < n) {                                                           // bounds check
        c[idx] = a[idx] + b[idx];                                            // elementwise sum
    }                                                                        // end bounds
}                                                                            // end kernel

__global__ void access_coalesced(const float* in, float* out, int n) {         // Task 3: coalesced access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          // global thread index
    if (idx < n) {                                                           // bounds check
        out[idx] = in[idx] * 2.0f + 1.0f;                                    // contiguous read/write
    }                                                                        // end bounds
}                                                                            // end kernel

__global__ void access_noncoalesced(const float* in, float* out, int n) {      // Task 3: non-coalesced access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                          // global thread index
    const int BAD_STRIDE = 1024;                                             // stride that breaks locality
    int j = (idx * BAD_STRIDE) % n;                                          // scattered index mapping
    if (idx < n) {                                                           // bounds check
        out[idx] = in[j] * 2.0f + 1.0f;                                      // scattered read, linear write
    }                                                                        // end bounds
}                                                                            // end kernel

static float time_ms_scale_global(float* d, float k, int n, int grid, int block, int iters) { // timing helper for Task 1 global
    cudaEvent_t start, stop;                                                 // CUDA events
    CUDA_CHECK(cudaEventCreate(&start));                                     // create start event
    CUDA_CHECK(cudaEventCreate(&stop));                                      // create stop event
    for (int i = 0; i < 5; ++i) {                                            // warmup launches
        scale_global<<<grid, block>>>(d, k, n);                              // warmup kernel
    }                                                                        // end warmup
    CUDA_CHECK(cudaDeviceSynchronize());                                     // wait for warmup
    CUDA_CHECK(cudaEventRecord(start));                                      // start timer
    for (int i = 0; i < iters; ++i) {                                        // measured launches
        scale_global<<<grid, block>>>(d, k, n);                              // measured kernel
    }                                                                        // end measured
    CUDA_CHECK(cudaEventRecord(stop));                                       // stop timer
    CUDA_CHECK(cudaEventSynchronize(stop));                                  // wait for stop
    float ms = 0.0f;                                                         // elapsed time storage
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));                      // compute elapsed ms
    CUDA_CHECK(cudaEventDestroy(start));                                     // destroy start event
    CUDA_CHECK(cudaEventDestroy(stop));                                      // destroy stop event
    return ms / iters;                                                       // average ms per launch
}                                                                            // end helper

static float time_ms_scale_shared(float* d, float k, int n, int grid, int block, int iters) { // timing helper for Task 1 shared
    cudaEvent_t start, stop;                                                 // CUDA events
    CUDA_CHECK(cudaEventCreate(&start));                                     // create start event
    CUDA_CHECK(cudaEventCreate(&stop));                                      // create stop event
    size_t shmem = static_cast<size_t>(block) * sizeof(float);               // dynamic shared memory size
    for (int i = 0; i < 5; ++i) {                                            // warmup launches
        scale_shared<<<grid, block, shmem>>>(d, k, n);                       // warmup kernel
    }                                                                        // end warmup
    CUDA_CHECK(cudaDeviceSynchronize());                                     // wait for warmup
    CUDA_CHECK(cudaEventRecord(start));                                      // start timer
    for (int i = 0; i < iters; ++i) {                                        // measured launches
        scale_shared<<<grid, block, shmem>>>(d, k, n);                       // measured kernel
    }                                                                        // end measured
    CUDA_CHECK(cudaEventRecord(stop));                                       // stop timer
    CUDA_CHECK(cudaEventSynchronize(stop));                                  // wait for stop
    float ms = 0.0f;                                                         // elapsed time storage
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));                      // compute elapsed ms
    CUDA_CHECK(cudaEventDestroy(start));                                     // destroy start event
    CUDA_CHECK(cudaEventDestroy(stop));                                      // destroy stop event
    return ms / iters;                                                       // average ms per launch
}                                                                            // end helper

static float time_ms_vec_add(const float* a, const float* b, float* c, int n, int grid, int block, int iters) { // timing helper for add
    cudaEvent_t start, stop;                                                 // CUDA events
    CUDA_CHECK(cudaEventCreate(&start));                                     // create start event
    CUDA_CHECK(cudaEventCreate(&stop));                                      // create stop event
    for (int i = 0; i < 5; ++i) {                                            // warmup launches
        vec_add<<<grid, block>>>(a, b, c, n);                                // warmup kernel
    }                                                                        // end warmup
    CUDA_CHECK(cudaDeviceSynchronize());                                     // wait for warmup
    CUDA_CHECK(cudaEventRecord(start));                                      // start timer
    for (int i = 0; i < iters; ++i) {                                        // measured launches
        vec_add<<<grid, block>>>(a, b, c, n);                                // measured kernel
    }                                                                        // end measured
    CUDA_CHECK(cudaEventRecord(stop));                                       // stop timer
    CUDA_CHECK(cudaEventSynchronize(stop));                                  // wait for stop
    float ms = 0.0f;                                                         // elapsed time storage
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));                      // compute elapsed ms
    CUDA_CHECK(cudaEventDestroy(start));                                     // destroy start event
    CUDA_CHECK(cudaEventDestroy(stop));                                      // destroy stop event
    return ms / iters;                                                       // average ms per launch
}                                                                            // end helper

static float time_ms_access_coalesced(const float* in, float* out, int n, int grid, int block, int iters) { // timing helper for coalesced
    cudaEvent_t start, stop;                                                 // CUDA events
    CUDA_CHECK(cudaEventCreate(&start));                                     // create start event
    CUDA_CHECK(cudaEventCreate(&stop));                                      // create stop event
    for (int i = 0; i < 5; ++i) {                                            // warmup launches
        access_coalesced<<<grid, block>>>(in, out, n);                       // warmup kernel
    }                                                                        // end warmup
    CUDA_CHECK(cudaDeviceSynchronize());                                     // wait for warmup
    CUDA_CHECK(cudaEventRecord(start));                                      // start timer
    for (int i = 0; i < iters; ++i) {                                        // measured launches
        access_coalesced<<<grid, block>>>(in, out, n);                       // measured kernel
    }                                                                        // end measured
    CUDA_CHECK(cudaEventRecord(stop));                                       // stop timer
    CUDA_CHECK(cudaEventSynchronize(stop));                                  // wait for stop
    float ms = 0.0f;                                                         // elapsed time storage
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));                      // compute elapsed ms
    CUDA_CHECK(cudaEventDestroy(start));                                     // destroy start event
    CUDA_CHECK(cudaEventDestroy(stop));                                      // destroy stop event
    return ms / iters;                                                       // average ms per launch
}                                                                            // end helper

static float time_ms_access_noncoalesced(const float* in, float* out, int n, int grid, int block, int iters) { // timing helper for non-coalesced
    cudaEvent_t start, stop;                                                 // CUDA events
    CUDA_CHECK(cudaEventCreate(&start));                                     // create start event
    CUDA_CHECK(cudaEventCreate(&stop));                                      // create stop event
    for (int i = 0; i < 5; ++i) {                                            // warmup launches
        access_noncoalesced<<<grid, block>>>(in, out, n);                    // warmup kernel
    }                                                                        // end warmup
    CUDA_CHECK(cudaDeviceSynchronize());                                     // wait for warmup
    CUDA_CHECK(cudaEventRecord(start));                                      // start timer
    for (int i = 0; i < iters; ++i) {                                        // measured launches
        access_noncoalesced<<<grid, block>>>(in, out, n);                    // measured kernel
    }                                                                        // end measured
    CUDA_CHECK(cudaEventRecord(stop));                                       // stop timer
    CUDA_CHECK(cudaEventSynchronize(stop));                                  // wait for stop
    float ms = 0.0f;                                                         // elapsed time storage
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));                      // compute elapsed ms
    CUDA_CHECK(cudaEventDestroy(start));                                     // destroy start event
    CUDA_CHECK(cudaEventDestroy(stop));                                      // destroy stop event
    return ms / iters;                                                       // average ms per launch
}                                                                            // end helper

static int grid_for(int n, int block) {                                      // simple grid size calculator
    return (n + block - 1) / block;                                          // one thread per element mapping
}                                                                            // end helper

int main() {                                                                 // program entry point
    std::cout << std::fixed << std::setprecision(4);                         // stable numeric formatting
    std::vector<float> h_a(N);                                               // host array A
    std::vector<float> h_b(N);                                               // host array B
    for (int i = 0; i < N; ++i) {                                            // initialize host data
        h_a[i] = 0.001f * static_cast<float>(i % 1000);                      // pattern for A
        h_b[i] = 0.002f * static_cast<float>(i % 1000);                      // pattern for B
    }                                                                        // end init loop
    float* d_a = nullptr;                                                    // device pointer A
    float* d_b = nullptr;                                                    // device pointer B
    float* d_c = nullptr;                                                    // device pointer C
    CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(float)));                         // allocate device A
    CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(float)));                         // allocate device B
    CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(float)));                         // allocate device C
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice)); // copy A to GPU
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), N * sizeof(float), cudaMemcpyHostToDevice)); // copy B to GPU
    CUDA_CHECK(cudaMemset(d_c, 0, N * sizeof(float)));                       // clear C on GPU
    std::cout << "N = " << N << " elements\n";                               // print N
    std::cout << "\nTask 1: scale array (global vs shared)\n";               // Task 1 title
    int block1 = 256;                                                        // chosen block size for Task 1
    int grid1 = grid_for(N, block1);                                         // grid size for Task 1
    int iters = 50;                                                          // iterations for timing
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice)); // reset A
    float t1_global = time_ms_scale_global(d_a, SCALE_K, N, grid1, block1, iters);     // measure global
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), N * sizeof(float), cudaMemcpyHostToDevice)); // reset A
    float t1_shared = time_ms_scale_shared(d_a, SCALE_K, N, grid1, block1, iters);     // measure shared
    std::cout << "block=" << block1 << " grid=" << grid1 << "\n";            // print config
    std::cout << "global: " << t1_global << " ms\n";                         // print time global
    std::cout << "shared: " << t1_shared << " ms\n";                         // print time shared
    std::cout << "\nTask 2: vector add (block size impact)\n";               // Task 2 title
    int blocks2[3] = {128, 256, 512};                                        // required block sizes
    for (int k = 0; k < 3; ++k) {                                            // loop over block sizes
        int bsz = blocks2[k];                                                // current block size
        int gsz = grid_for(N, bsz);                                          // grid for current block size
        CUDA_CHECK(cudaMemset(d_c, 0, N * sizeof(float)));                   // clear output
        float t = time_ms_vec_add(d_a, d_b, d_c, N, gsz, bsz, iters);         // measure add time
        std::cout << "block=" << std::setw(4) << bsz                         // print block size
                  << " grid=" << std::setw(6) << gsz                         // print grid size
                  << " time=" << t << " ms\n";                               // print time
    }                                                                        // end loop
    std::cout << "\nTask 3: global memory access (coalesced vs non-coalesced)\n"; // Task 3 title
    int block3 = 256;                                                        // fixed block size for Task 3
    int grid3 = grid_for(N, block3);                                         // grid size for Task 3
    CUDA_CHECK(cudaMemset(d_c, 0, N * sizeof(float)));                       // clear output
    float t3_coal = time_ms_access_coalesced(d_a, d_c, N, grid3, block3, iters);       // measure coalesced
    CUDA_CHECK(cudaMemset(d_c, 0, N * sizeof(float)));                       // clear output again
    float t3_non = time_ms_access_noncoalesced(d_a, d_c, N, grid3, block3, iters);     // measure non-coalesced
    std::cout << "block=" << block3 << " grid=" << grid3 << "\n";            // print config
    std::cout << "coalesced:     " << t3_coal << " ms\n";                    // print coalesced time
    std::cout << "non-coalesced: " << t3_non << " ms\n";                     // print non-coalesced time
    std::cout << "\nTask 4: tune configuration (vector add)\n";              // Task 4 title
    int badBlock = 32;                                                       // intentionally weak config (few threads)
    int badGrid = grid_for(N, badBlock);                                     // grid for weak config
    CUDA_CHECK(cudaMemset(d_c, 0, N * sizeof(float)));                       // clear output
    float t_bad = time_ms_vec_add(d_a, d_b, d_c, N, badGrid, badBlock, iters);// measure weak config
    int bestBlock = 256;                                                     // typical strong config
    int bestGrid = grid_for(N, bestBlock);                                   // grid for strong config
    CUDA_CHECK(cudaMemset(d_c, 0, N * sizeof(float)));                       // clear output
    float t_best = time_ms_vec_add(d_a, d_b, d_c, N, bestGrid, bestBlock, iters); // measure strong config
    std::cout << "non-opt block=" << badBlock << " grid=" << badGrid << " time=" << t_bad << " ms\n";  // print weak
    std::cout << "opt     block=" << bestBlock << " grid=" << bestGrid << " time=" << t_best << " ms\n"; // print strong
    CUDA_CHECK(cudaFree(d_a));                                               // free device A
    CUDA_CHECK(cudaFree(d_b));                                               // free device B
    CUDA_CHECK(cudaFree(d_c));                                               // free device C
    CUDA_CHECK(cudaDeviceReset());                                           // reset device for clean exit
    std::cout << "\nDone.\n";                                                // finish message
    return 0;                                                                // success code
}                                                                            // end main
