#ifndef UTILS_SEARCH_CUH
#define UTILS_SEARCH_CUH

#include <cstdint>


template<typename T>
__forceinline__ __device__ uint32_t lower_bound(T* array, uint32_t size, const T& v)
{
    if (array == nullptr || size == 0 || array[size - 1] < v) return UINT32_MAX;

    uint32_t low = 0u, high = size - 1, mid = (low + high) / 2;
    while (low < high)
    {
        if (array[mid] < v)
        {
            low = mid + 1;
        }
        else
        {
            high = mid;
        }
        mid = (low + high) / 2;
    }
    return mid;
}

template<typename T>
__forceinline__ __device__ uint32_t linear_search(T* array, uint32_t size, const T& v)
{
    if (array == nullptr || size == 0) return UINT32_MAX;

    uint32_t mid = 0u;

    while (mid < size)
    {
        if (array[mid] == v)
        {
            return mid;
        }
        mid++;
    }
    return UINT32_MAX;
}


template<typename T>
__forceinline__ __device__ uint32_t linear_search_flag(T* array, uint32_t size, uint32_t *flag, const T& v)
{
    if (array == nullptr || size == 0) return UINT32_MAX;

    uint32_t mid = 0u;

    while (mid < size)
    {
        if (array[mid] == v && flag[mid] == 0)
        {
            return mid;
        }
        mid++;
    }
    return UINT32_MAX;
}


#endif //UTILS_SEARCH_CUH
