#ifndef UTILS_CUDAMEM_H
#define UTILS_CUDAMEM_H


#include <cstdint>


/*struct PoolElem
{
private:
    size_t start_pos_on_pool_;
public:
    PoolElem(size_t start_pos_on_pool);
    __device__ uint32_t& operator[] (size_t i);
};*/

#define PoolElem unsigned long long int

struct MemPool
{
    uint32_t *array_;
    size_t capability_;
    size_t available_start_;
    size_t available_end_;

    MemPool();
    void Alloc(size_t capability);
    void Free();
    PoolElem TryMax();
    size_t GetFree();
    void Push(size_t size);
    void Pop(size_t size);
};

extern __constant__ MemPool C_MEMPOOL;

#endif //UTILS_CUDAMEM_H
