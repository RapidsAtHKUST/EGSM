#include <cstdint>

#include "utils/cuda_helpers.h"
#include "utils/mem_pool.h"


__constant__ MemPool C_MEMPOOL;

/*PoolElem::PoolElem(size_t start_pos_on_pool)
: start_pos_on_pool_(start_pos_on_pool)
{}

__device__ uint32_t& PoolElem::operator[] (size_t i)
{
    return C_MEMPOOL.array_[(start_pos_on_pool_ + i) % C_MEMPOOL.capability_];
}*/


MemPool::MemPool()
{}

void MemPool::Alloc(size_t capability)
{
    cudaErrorCheck(cudaMalloc(&array_, sizeof(uint32_t) * capability));
    capability_ = capability;
    available_start_ = 0;
    available_end_ = capability;
}

void MemPool::Free()
{
    cudaErrorCheck(cudaFree(array_));
}

PoolElem MemPool::TryMax()
{
    //PoolElem elem(available_start_);
    return available_start_;
}


size_t MemPool::GetFree()
{
    return (available_end_ + capability_ - available_start_) % capability_;
}

void MemPool::Push(size_t size)
{
    available_start_ = (available_start_ + size) % capability_;
}

void MemPool::Pop(size_t size)
{
    available_end_ = (available_end_ + size) % capability_;
}
