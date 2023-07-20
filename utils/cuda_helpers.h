#ifndef UTILS_CUDA_HELPERS_H
#define UTILS_CUDA_HELPERS_H

#include <stdio.h>
#include <cstdint>


#define DIV_CEIL(a,b) ((a) / (b) + ((a) % (b) != 0))

#define TO_GB(x) (x / 1024.0 / 1024.0 / 1024.0)

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#define cudaErrorCheckInKernel(call)                            \
{                                                               \
    cudaError_t cucheck_err = (call);                           \
    if(cucheck_err != cudaSuccess) {                            \
        const char *err_str = cudaGetErrorString(cucheck_err);  \
        printf("%s (%d): %s\n", __FILE__, __LINE__, err_str);   \
        assert(0);                                              \
    }                                                           \
}

// Recording Time

#define TIME_INIT() cudaEvent_t gpu_start, gpu_end;             \
    float kernel_time, total_kernel = 0.0, total_host = 0.0;    \
    auto cpu_start = std::chrono::high_resolution_clock::now(); \
    auto cpu_end = std::chrono::high_resolution_clock::now();   \
    std::chrono::duration<double> diff = cpu_end - cpu_start;

#define TIME_START()                                            \
    cpu_start = std::chrono::high_resolution_clock::now();      \
    cudaEventCreate(&gpu_start);                                \
    cudaEventCreate(&gpu_end);                                  \
    cudaEventRecord(gpu_start)

#define TIME_END()                                              \
    cpu_end = std::chrono::high_resolution_clock::now();        \
    cudaEventRecord(gpu_end);                                   \
    cudaEventSynchronize(gpu_start);                            \
    cudaEventSynchronize(gpu_end);                              \
    cudaEventElapsedTime(&kernel_time, gpu_start, gpu_end);     \
    total_kernel += kernel_time;                                \
    diff = cpu_end - cpu_start;                                 \
    total_host += diff.count();

#define PRINT_LOCAL_TIME(name)                                  \
    std::cout << name << ", time (ms): "                        \
    << static_cast<unsigned long>(diff.count() * 1000)          \
    << "(host), "                                               \
    << static_cast<unsigned long>(kernel_time)                  \
    << "(kernel)\n"

#define PRINT_TOTAL_TIME(name)                                  \
    std::cout << name << " time (ms): "                        \
    << static_cast<unsigned long>(total_host * 1000)            \
    << "(host) "                                               \
    << static_cast<unsigned long>(total_kernel)                 \
    << "(kernel)\n";

// Recording Memory Space

#define MEM_INIT() size_t mf, ma;

#define PRINT_MEM_INFO(name)                                    \
    cudaMemGetInfo(&mf, &ma);                                   \
    std::cout << name << ", Free " << TO_GB(mf) <<              \
        "GB, Total: " << TO_GB(ma) << "GB\n";

#endif //UTILS_CUDA_HELPERS_H
