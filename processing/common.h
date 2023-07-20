#ifndef PROCESSING_COMMON_H
#define PROCESSING_COMMON_H


#include <cstdint>
#include "utils/search.cuh"
#include "utils/globals.h"
#include "graph/graph.h"
#include "processing/plan.h"


__forceinline__ __device__ void copyUtilsConstantToShared(GraphUtils &s_utils)
{
    // copy data from constant memory to shared memory
    for (uint32_t i = threadIdx.x; i < MAX_VCOUNT * MAX_VCOUNT; i += blockDim.x)
    {
        s_utils.eidx_[i] = C_UTILS.eidx_[i];
    }
    for (uint32_t i = threadIdx.x; i < MAX_VCOUNT; i += blockDim.x)
    {
        s_utils.nbrbits_[i] = C_UTILS.nbrbits_[i];
    }
    __syncthreads();
}


__forceinline__ __device__ void buildMappedVs(
    uint16_t& mapped_vs, 
    uint8_t order[MAX_VCOUNT],
    uint32_t lane_id
) {
    uint16_t bitmap = 0u;
    if (lane_id < C_NUM_VQ)
    {
        uint8_t cur_order = order[lane_id];
        if (cur_order != UINT8_MAX)
        {
            bitmap = (1 << cur_order);
        }
    }
    __syncwarp();
    // reduce bitmap to get mapped_vs
    for (uint32_t i = 1; i < WARP_SIZE; i *= 2)
    {
        bitmap += __shfl_down_sync(0xffffffff, bitmap, i);
    }
    __syncwarp();
    if (lane_id == 0)
    {
        mapped_vs = bitmap;
    }
    __syncwarp();
}

__forceinline__ __device__ void buildMappedVs(
    uint16_t& mapped_vs, 
    InitialOrder initial_order,
    uint32_t lane_id
) {
    uint16_t bitmap = 0u;
    if (lane_id < C_NUM_VQ)
    {
        uint8_t cur_order = initial_order.u[lane_id];
        if (cur_order != UINT8_MAX)
        {
            bitmap = (1 << cur_order);
        }
    }
    __syncwarp();
    // reduce bitmap to get mapped_vs
    for (uint32_t i = 1; i < WARP_SIZE; i *= 2)
    {
        bitmap += __shfl_down_sync(0xffffffff, bitmap, i);
    }
    __syncwarp();
    if (lane_id == 0)
    {
        mapped_vs = bitmap;
    }
    __syncwarp();
}


__forceinline__ __device__ bool gpuIntersection(
    uint32_t **arrays,
    uint32_t *sizes,
    const uint32_t *min_array,
    uint32_t min_off,
    uint32_t start,
    uint32_t offset1,
    uint32_t offset2,
    uint32_t lane_id
) {
    if (start + lane_id >= sizes[min_off]) return false;

    uint32_t element_of_lane = min_array[start + lane_id];

    for (uint32_t i = offset1; i < offset2; i++)
    {
        if (i == min_off) continue;
        if (sizes[i] == UINT32_MAX)
        {
            uint2 res = tries.HashSearch(i, element_of_lane);
            if (res.x == UINT32_MAX)
            {
                return false;
            }
        }
        else
        {
            uint32_t res = lower_bound(arrays[i], sizes[i], element_of_lane);
            if (res == UINT32_MAX || arrays[i][res] != element_of_lane)
            {
                return false;
            }
        }
    }
    return true;
}

#endif //PROCESSING_COMMON_H
