#ifndef PROCESSING_JOIN_BFS_DFS_H
#define PROCESSING_JOIN_BFS_DFS_H

#include <cstdint>
#include <cooperative_groups.h>
#include <cuda/std/type_traits>

#include "utils/types.h"
#include "utils/mem_pool.h"
#include "processing/plan.h"
#include "processing/common.h"


__forceinline__ __device__ void set_next_u(
    uint16_t& mapped_vs,
    uint32_t start[MAX_VCOUNT],
    uint32_t intersection_input_sizes[MAX_ECOUNT * 2],
    uint32_t tries_vsizes[MAX_ECOUNT * 2],
    uint8_t& next_order,
    uint8_t min_offs[MAX_VCOUNT],
    int8_t& mask_index,
    const GraphUtils& s_utils,
    uint32_t warp_id,
    uint32_t lane_id
) {
    // 1. get all extendable vertices
    if (lane_id == 0)
    {
        while (((~mapped_vs) & c_plan.masks_[mask_index]) == 0)
        {
            mask_index ++;
            //printf("mask %d\n", mask_index);
        }
    }
    __syncwarp();
    bool extendable = false, other_extendable;
    uint32_t can_size = UINT32_MAX, other_can_size;
    uint8_t local_min_offs = UINT8_MAX, other_local_min_offs;
    uint8_t select_v = lane_id, other_select;

    uint16_t nbr_bits;
    uint8_t local_nbr, local_offs;
    if (lane_id < C_NUM_VQ)
    {
        extendable = (
            s_utils.nbrbits_[lane_id] & mapped_vs) && 
            (~mapped_vs & (1 << lane_id)) && 
            (c_plan.masks_[mask_index] & (1 << lane_id)
        );
    }
    __syncwarp();
    // 2. get the estimated candidate set size of each extenable vertex
    if (extendable)
    {
        nbr_bits = s_utils.nbrbits_[lane_id] & mapped_vs;
        while (nbr_bits)
        {
            // iterate over each mapped vertex
            local_nbr = __ffs(nbr_bits) - 1;
#ifdef DEBUG
            printf("lane_id=%d, nbr_bits=%04x, local_nbr=%u\n", 
                lane_id, nbr_bits, local_nbr);
#endif
            nbr_bits &= (~(1 << local_nbr));
            local_offs = s_utils.eidx_[lane_id * C_NUM_VQ + local_nbr];
#ifdef DEBUG
            printf("lane_id=%d, local_offs=%d\n", lane_id, local_offs);
#endif

            if (intersection_input_sizes[local_offs] < can_size)
            {
                can_size = intersection_input_sizes[local_offs];
                local_min_offs = local_offs;
#ifdef DEBUG
                printf("lane_id=%d, local_min_offs=%d, can_size=%d\n", 
                    lane_id, local_min_offs, can_size);
#endif
            }
        }
        nbr_bits = s_utils.nbrbits_[lane_id] & (~mapped_vs);
        while (nbr_bits)
        {
            // iterate over each unmapped vertex
            local_nbr = __ffs(nbr_bits) - 1;
#ifdef DEBUG
            printf("lane_id=%d, nbr_bits=%04x, local_nbr=%u\n", 
                lane_id, nbr_bits, local_nbr);
#endif
            nbr_bits &= (~(1 << local_nbr));
            local_offs = s_utils.eidx_[lane_id * C_NUM_VQ + local_nbr];
#ifdef DEBUG
            printf("lane_id=%d, local_offs=%d\n", lane_id, local_offs);
#endif

            if (tries_vsizes[local_offs] < can_size)
            {
                can_size = tries_vsizes[local_offs];
                local_min_offs = local_offs;
#ifdef DEBUG
                printf("lane_id=%d, local_min_offs=%d, can_size=%d\n", 
                    lane_id, local_min_offs, can_size);
#endif
            }
        }
    }
    __syncwarp();
    if (!C_ADAPTIVE_ORDERING)
    {
        can_size = __popc(s_utils.nbrbits_[lane_id] & mapped_vs);
    }
    // 3. get the vertex with minimum candidate set size
    for (uint32_t i = 1; i < WARP_SIZE; i *= 2)
    {
        other_extendable = __shfl_down_sync(0xffffffff, extendable, i);
        other_can_size = __shfl_down_sync(0xffffffff, can_size, i);
        other_local_min_offs = __shfl_down_sync(0xffffffff, local_min_offs, i);
        other_select = __shfl_down_sync(0xffffffff, select_v, i);
        if (lane_id % (2 * i) == 0)
        {
            if (!other_extendable) continue;
            else if (C_ADAPTIVE_ORDERING)
            {
                if ((!extendable) || (can_size > other_can_size))
                {
                    extendable = other_extendable;
                    select_v = other_select;
                    can_size = other_can_size;
                    local_min_offs = other_local_min_offs;
#ifdef DEBUG
                    printf("lane_id=%d, round=%d, exchange, new_select_v=%d\n",
                        lane_id, i, select_v);
#endif
                }
            }
            else
            {
                if ((!extendable) || (can_size < other_can_size))
                {
                    extendable = other_extendable;
                    select_v = other_select;
                    can_size = other_can_size;
                    local_min_offs = other_local_min_offs;
#ifdef DEBUG
                    printf("lane_id=%d, round=%d, exchange, new_select_v=%d\n",
                        lane_id, i, select_v);
#endif
                }
            }
        }
    }
    __syncwarp();
    if (lane_id == 0)
    {
        next_order = select_v;
        mapped_vs |= (1 << select_v);
        min_offs[select_v] = local_min_offs;
#ifdef DEBUG
        printf("select next vertex=%d, local_min_offs=%d, mapped_vs=%x\n", 
            select_v, local_min_offs, mapped_vs);
#endif
        if (intersection_input_sizes[min_offs[select_v]] == UINT32_MAX)
        {
            // if this array is the first level of any trie
            intersection_input_sizes[min_offs[select_v]] = 
                tries_vsizes[min_offs[select_v]];
            start[select_v] = 0u;
        }
    }
    __syncwarp();
}


__forceinline__ __device__ void map_new_v(
    uint32_t u, 
    uint32_t v, 
    uint16_t mapped_vs,
    uint32_t *intersection_input[MAX_ECOUNT * 2],
    uint32_t intersection_input_sizes[MAX_ECOUNT * 2],
    const GraphUtils& s_utils,
    uint32_t warp_id,
    uint32_t lane_id
) {
    if (
        (lane_id < MAX_VCOUNT) && 
        ((1 << lane_id) & (~mapped_vs) & s_utils.nbrbits_[u]))
    {
        // find the neighbor array
        uint32_t local_off = s_utils.eidx_[u * C_NUM_VQ + lane_id];
        uint32_t reversed_local_off = s_utils.eidx_[lane_id * C_NUM_VQ + u];
        uint2 res = tries.HashSearch(local_off, v);
#ifdef DEBUG
        printf("lane_id=%u, v=%u, local_off=%u, reversed=%u, res=(%u,%u)\n",
            lane_id, v, local_off, reversed_local_off, res.x, res.y);
#endif

        intersection_input_sizes[reversed_local_off] = 
            res.x == UINT32_MAX ?
            0u : tries.values_[res.y][
                (tries.hash_table_offs_[local_off] + res.x) * 2 + 1
            ];
        intersection_input[reversed_local_off] =
            res.x == UINT32_MAX ?
            NULL : tries.neighbors_[res.y] + tries.values_[res.y][
                (tries.hash_table_offs_[local_off] + res.x) * 2
            ];
#ifdef DEBUG
        printf("lane_id=%u, size=%u, intersection_input=%u\n", 
            lane_id, intersection_input_sizes[reversed_local_off], 
            intersection_input[reversed_local_off][0]
        );
#endif
    }
    __syncwarp();
}

__forceinline__ __device__ void unmap_u(
    uint32_t u, 
    uint16_t mapped_vs,
    uint32_t *intersection_input[MAX_ECOUNT * 2],
    uint32_t intersection_input_sizes[MAX_ECOUNT * 2],
    const GraphUtils& s_utils,
    uint32_t warp_id,
    uint32_t lane_id,
    HashedTries& tries
) {
    if (
        (lane_id < 12) && 
        ((1 << lane_id) & (~mapped_vs) & s_utils.nbrbits_[u])
    ) {
        // find the neighbor array
        uint32_t reversed_local_off = s_utils.eidx_[lane_id * C_NUM_VQ + u];
#ifdef DEBUG
        printf("lane_id=%u, v=%u, local_off=%u, reversed=%u\n",
            lane_id, v, reversed_local_off);
#endif

        intersection_input_sizes[reversed_local_off] = UINT32_MAX;
        intersection_input[reversed_local_off] = NULL;
#ifdef DEBUG
        printf("lane_id=%u, size=%u, intersection_input=%u\n",
            lane_id, intersection_input_sizes[reversed_local_off],
            intersection_input[reversed_local_off][0]
        );
#endif
    }
}


__global__ void joinDFSGroupKernel(
    PoolElem res,
    unsigned long long int res_size,
    PoolElem new_res,
    unsigned long long int max_new_res_size,
    unsigned long long int *new_res_size,
    InitialOrder initial_order,
    uint8_t next_u,
    uint8_t next_u_min_off,
    uint8_t max_depth,
    bool only_one_group,
    uint32_t *pending_count,
    uint32_t *lb_triggered
);

#endif //PROCESSING_JOIN_BFS_DFS_H
