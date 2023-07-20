#include <cstdint>
#include <cooperative_groups.h>

#include "utils/config.h"
#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "utils/search.cuh"

#include "processing/plan.h"
#include "structures/hashed_tries.h"
#include "structures/hashed_trie_manager.h"
#include "processing/common.h"
#include "processing/plan.h"


__global__ void joinBFSFirstCount(
    uint32_t first_u, uint32_t first_u_min_off,
    uint32_t *new_partial_result_sizes
) {
    __shared__ uint32_t tries_vsizes[MAX_ECOUNT * 2];

    // copy from constant memory to shared memory
    for (uint32_t i = threadIdx.x; i < C_NUM_EQ * 2; i += blockDim.x)
    {
        tries_vsizes[i] = tries.compacted_vs_sizes_[i];
    }
    __syncthreads();

    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t global_warp_id = tid / WARP_SIZE;
    uint32_t lane_id = tid % WARP_SIZE;

    if (tid >= tries_vsizes[first_u_min_off])
    {
        return;
    }
    if (lane_id == 0)
    {
        new_partial_result_sizes[global_warp_id] = 0u;
    }
    __syncwarp();

    uint32_t v = tries.compacted_vs_[first_u_min_off * C_MAX_L_FREQ + tid];
    bool found = true;
    for (uint32_t i = C_ORDER_OFFS[first_u]; i < C_ORDER_OFFS[first_u + 1]; i++)
    {
        if (i == first_u_min_off) continue;
        uint2 res = tries.HashSearch(i, v);
        if (res.x == UINT32_MAX)
        {
            found = false;
        }
    }
    __syncwarp(__activemask());
    if (found)
    {
        auto group = cooperative_groups::coalesced_threads();
        auto rank = group.thread_rank();
        if (rank == 0)
        {
            new_partial_result_sizes[global_warp_id] = group.size();
        }
    }
}

__global__ void joinBFSFirstWrite(
    uint32_t first_u,
    uint32_t first_u_min_off,
    uint32_t *new_partial_result_sizes_prefix_sum,
    uint32_t *new_partial_results
) {
    __shared__ uint32_t tries_vsizes[MAX_ECOUNT * 2];

    // copy from constant memory to shared memory
    for (uint32_t i = threadIdx.x; i < C_NUM_EQ * 2; i += blockDim.x)
    {
        tries_vsizes[i] = tries.compacted_vs_sizes_[i];
    }
    __syncthreads();

    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t global_warp_id = tid / WARP_SIZE;

    if (tid >= tries_vsizes[first_u_min_off])
    {
        return;
    }
    uint32_t v = tries.compacted_vs_[first_u_min_off * C_MAX_L_FREQ + tid];
    bool found = true;
    for (uint32_t i = C_ORDER_OFFS[first_u]; i < C_ORDER_OFFS[first_u + 1]; i++)
    {
        if (i == first_u_min_off) continue;
        uint2 res = tries.HashSearch(i, v);
        if (res.x == UINT32_MAX)
        {
            found = false;
        }
    }
    __syncwarp(__activemask());
    if (found)
    {
        auto group = cooperative_groups::coalesced_threads();
        auto rank = group.thread_rank();
        new_partial_results[new_partial_result_sizes_prefix_sum[global_warp_id] + rank] = v;
    }
}


__global__ void bfsRemainingCount(
    uint32_t *results,
    uint32_t result_size,
    uint32_t *new_result_sizes_per_warp,
    InitialOrder initial_order,
    uint8_t next_u,
    uint8_t depth
) {
    __shared__ uint32_t partial_result[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ uint32_t local_size[WARP_PER_BLOCK];
    __shared__ InitialOrder order[WARP_PER_BLOCK];

    __shared__ uint32_t *intersection_input[WARP_PER_BLOCK][MAX_ECOUNT * 2];
    __shared__ uint32_t intersection_input_sizes[WARP_PER_BLOCK][MAX_ECOUNT * 2];
    __shared__ uint8_t next_u_min_off[WARP_PER_BLOCK];
    __shared__ uint16_t mapped_vs[WARP_PER_BLOCK];

    __shared__ GraphUtils s_utils;
    __shared__ uint32_t tries_vsizes[MAX_ECOUNT * 2];

    copyUtilsConstantToShared(s_utils);
    // copy from constant memory to shared memory
    for (uint32_t i = threadIdx.x; i < C_NUM_EQ * 2; i += blockDim.x)
    {
        tries_vsizes[i] = tries.compacted_vs_sizes_[i];
    }
    __syncthreads();

    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t global_warp_id = tid / WARP_SIZE;
    uint32_t lane_id = tid % WARP_SIZE;

    if (global_warp_id >= result_size)
    {
        return;
    }

    if (lane_id < MAX_VCOUNT)
    {
        order[warp_id].u[lane_id] = initial_order.u[lane_id];
    }
    __syncwarp();

    // initialize shared memory
    for (uint32_t i = lane_id; i < MAX_ECOUNT * 2; i += WARP_SIZE)
    {
        intersection_input[warp_id][i] = NULL;
        intersection_input_sizes[warp_id][i] = UINT32_MAX;
    }
    __syncwarp();
    if (lane_id == 0)
    {
        new_result_sizes_per_warp[global_warp_id] = 0u;
        local_size[warp_id] = 0u;
    }
    __syncwarp();

    buildMappedVs(mapped_vs[warp_id], order[warp_id], lane_id);

    if (lane_id < depth)
    {
        partial_result[warp_id][order[warp_id].u[lane_id]] = results[global_warp_id * depth + lane_id];
    }
    __syncwarp();

    // a warp set the data for the intersection
    if (lane_id < C_NUM_VQ)
    {
        bool mapped_nbr = (mapped_vs[warp_id] & (1 << lane_id)) & s_utils.nbrbits_[next_u];
        bool unmapped_nbr = ((~mapped_vs[warp_id]) & (1 << lane_id)) & s_utils.nbrbits_[next_u];

        if (mapped_nbr || unmapped_nbr)
        {
            uint32_t local_off = s_utils.eidx_[lane_id * C_NUM_VQ + next_u];
            uint32_t reversed_local_off = s_utils.eidx_[next_u * C_NUM_VQ + lane_id];

            uint2 res = mapped_nbr ?
                tries.HashSearch(local_off, partial_result[warp_id][lane_id]) :
                make_uint2(UINT32_MAX, UINT32_MAX);
            
            intersection_input_sizes[warp_id][reversed_local_off] = 
                mapped_nbr ?
                (
                    res.x == UINT32_MAX ?
                    0u : tries.values_[res.y][(tries.hash_table_offs_[local_off] + res.x) * 2 + 1]
                ) :
                UINT32_MAX;
            intersection_input[warp_id][reversed_local_off] = 
                mapped_nbr ?
                (
                    res.x == UINT32_MAX ?
                    NULL : tries.neighbors_[res.y] + tries.values_[res.y][(tries.hash_table_offs_[local_off] + res.x) * 2]
                ) :
                NULL;
        }
    }
    __syncwarp();

    // a warp gets the shortes array for the intersection
    uint32_t selected_off = lane_id + C_ORDER_OFFS[next_u], other_selected_off;
    uint32_t size = 
        lane_id + C_ORDER_OFFS[next_u] < C_ORDER_OFFS[next_u + 1] ?
        (
            intersection_input_sizes[warp_id][lane_id + C_ORDER_OFFS[next_u]] != UINT32_MAX ?
            intersection_input_sizes[warp_id][lane_id + C_ORDER_OFFS[next_u]] :
            tries_vsizes[lane_id + C_ORDER_OFFS[next_u]]
        ) :
        UINT32_MAX;

    uint32_t other_size;
    for (uint32_t i = 1; i < WARP_SIZE; i *= 2)
    {
        other_selected_off = __shfl_down_sync(0xffffffff, selected_off, i);
        other_size = __shfl_down_sync(0xffffffff, size, i);
        if (lane_id % (2 * i) == 0)
        {
            if (other_size < size)
            {
                size = other_size;
                selected_off = other_selected_off;
            }
        }
    }
    __syncwarp();
    if (lane_id == 0)
    {
        next_u_min_off[warp_id] = selected_off;
        
        if (intersection_input_sizes[warp_id][next_u_min_off[warp_id]] == UINT32_MAX)
        {
            // if this array is the first level of any trie
            intersection_input_sizes[warp_id][next_u_min_off[warp_id]] = tries_vsizes[next_u_min_off[warp_id]];
        }
    }
    __syncwarp();

    // intersection
    for (uint32_t i = lane_id; i < intersection_input_sizes[warp_id][next_u_min_off[warp_id]]; i += WARP_SIZE)
    {
        uint32_t *min_array = intersection_input[warp_id][next_u_min_off[warp_id]] == NULL ?
            tries.compacted_vs_ + next_u_min_off[warp_id] * C_MAX_L_FREQ + i :
            intersection_input[warp_id][next_u_min_off[warp_id]] + i;
        uint32_t elem_index = i + lane_id;

        bool found = gpuIntersection(
            intersection_input[warp_id],
            intersection_input_sizes[warp_id],
            min_array,
            next_u_min_off[warp_id],
            i,
            C_ORDER_OFFS[next_u],
            C_ORDER_OFFS[next_u + 1],
            lane_id
        );
        if (found)
        {
            for (uint32_t j = 0; j < depth; j++)
            {
                if (partial_result[warp_id][order[warp_id].u[j]] == min_array[elem_index])
                {
                    found = false;
                    break;
                }
            }
        }
        __syncwarp(__activemask());
        if (found)
        {
            auto group = cooperative_groups::coalesced_threads();
            auto rank = group.thread_rank();
            if (rank == 0)
            {
                local_size[warp_id] += group.size();
            }
        }
    }
    __syncwarp();
    if (lane_id == 0)
    {
        new_result_sizes_per_warp[global_warp_id] = local_size[warp_id];
    }
}

__global__ void bfsRemainingWrite(
    uint32_t *results,
    uint32_t result_size,
    uint32_t *new_result_sizes_per_warp_prefix_sum,
    uint32_t *new_results,
    InitialOrder initial_order,
    uint8_t next_u,
    uint8_t depth
) {
    __shared__ uint32_t partial_result[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ uint32_t local_size[WARP_PER_BLOCK];
    __shared__ InitialOrder order[WARP_PER_BLOCK];

    __shared__ uint32_t *intersection_input[WARP_PER_BLOCK][MAX_ECOUNT * 2];
    __shared__ uint32_t intersection_input_sizes[WARP_PER_BLOCK][MAX_ECOUNT * 2];
    __shared__ uint8_t next_u_min_off[WARP_PER_BLOCK];
    __shared__ uint16_t mapped_vs[WARP_PER_BLOCK];

    __shared__ GraphUtils s_utils;
    __shared__ uint32_t tries_vsizes[MAX_ECOUNT * 2];

    copyUtilsConstantToShared(s_utils);
    // copy from constant memory to shared memory
    for (uint32_t i = threadIdx.x; i < C_NUM_EQ * 2; i += blockDim.x)
    {
        tries_vsizes[i] = tries.compacted_vs_sizes_[i];
    }
    __syncthreads();

    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t global_warp_id = tid / WARP_SIZE;
    uint32_t lane_id = tid % WARP_SIZE;

    if (global_warp_id >= result_size)
    {
        return;
    }

    if (lane_id < MAX_VCOUNT)
    {
        order[warp_id].u[lane_id] = initial_order.u[lane_id];
    }
    __syncwarp();

    // initialize shared memory
    for (uint32_t i = lane_id; i < MAX_ECOUNT * 2; i += WARP_SIZE)
    {
        intersection_input[warp_id][i] = NULL;
        intersection_input_sizes[warp_id][i] = UINT32_MAX;
    }
    __syncwarp();
    if (lane_id == 0)
    {
        local_size[warp_id] = 0u;
    }
    __syncwarp();

    buildMappedVs(mapped_vs[warp_id], order[warp_id], lane_id);

    if (lane_id < depth)
    {
        partial_result[warp_id][order[warp_id].u[lane_id]] = results[global_warp_id * depth + lane_id];
    }
    __syncwarp();

    // a warp set the data for the intersection
    if (lane_id < C_NUM_VQ)
    {
        bool mapped_nbr = (mapped_vs[warp_id] & (1 << lane_id)) & s_utils.nbrbits_[next_u];
        bool unmapped_nbr = ((~mapped_vs[warp_id]) & (1 << lane_id)) & s_utils.nbrbits_[next_u];

        if (mapped_nbr || unmapped_nbr)
        {
            uint32_t local_off = s_utils.eidx_[lane_id * C_NUM_VQ + next_u];
            uint32_t reversed_local_off = s_utils.eidx_[next_u * C_NUM_VQ + lane_id];

            uint2 res = mapped_nbr ?
                tries.HashSearch(local_off, partial_result[warp_id][lane_id]) :
                make_uint2(UINT32_MAX, UINT32_MAX);
            
            intersection_input_sizes[warp_id][reversed_local_off] = 
                mapped_nbr ?
                (
                    res.x == UINT32_MAX ?
                    0u : tries.values_[res.y][(tries.hash_table_offs_[local_off] + res.x) * 2 + 1]
                ) :
                UINT32_MAX;
            intersection_input[warp_id][reversed_local_off] = 
                mapped_nbr ?
                (
                    res.x == UINT32_MAX ?
                    NULL : tries.neighbors_[res.y] + tries.values_[res.y][(tries.hash_table_offs_[local_off] + res.x) * 2]
                ) :
                NULL;
        }
    }
    __syncwarp();

    // a warp gets the shortes array for the intersection
    uint32_t selected_off = lane_id + C_ORDER_OFFS[next_u], other_selected_off;
    uint32_t size = 
        lane_id + C_ORDER_OFFS[next_u] < C_ORDER_OFFS[next_u + 1] ?
        (
            intersection_input_sizes[warp_id][lane_id + C_ORDER_OFFS[next_u]] != UINT32_MAX ?
            intersection_input_sizes[warp_id][lane_id + C_ORDER_OFFS[next_u]] :
            tries_vsizes[lane_id + C_ORDER_OFFS[next_u]]
        ) :
        UINT32_MAX;

    uint32_t other_size;
    for (uint32_t i = 1; i < WARP_SIZE; i *= 2)
    {
        other_selected_off = __shfl_down_sync(0xffffffff, selected_off, i);
        other_size = __shfl_down_sync(0xffffffff, size, i);
        if (lane_id % (2 * i) == 0)
        {
            if (other_size < size)
            {
                size = other_size;
                selected_off = other_selected_off;
            }
        }
    }
    __syncwarp();
    if (lane_id == 0)
    {
        next_u_min_off[warp_id] = selected_off;
        
        if (intersection_input_sizes[warp_id][next_u_min_off[warp_id]] == UINT32_MAX)
        {
            // if this array is the first level of any trie
            intersection_input_sizes[warp_id][next_u_min_off[warp_id]] = tries_vsizes[next_u_min_off[warp_id]];
        }
    }
    __syncwarp();

    // intersection
    for (uint32_t i = 0; i < intersection_input_sizes[warp_id][next_u_min_off[warp_id]]; i += WARP_SIZE)
    {
        uint32_t *min_array = intersection_input[warp_id][next_u_min_off[warp_id]] == NULL ?
            tries.compacted_vs_ + next_u_min_off[warp_id] * C_MAX_L_FREQ + i :
            intersection_input[warp_id][next_u_min_off[warp_id]] + i;
        uint32_t elem_index = i + lane_id;

        bool found = gpuIntersection(
            intersection_input[warp_id],
            intersection_input_sizes[warp_id],
            min_array,
            next_u_min_off[warp_id],
            i,
            C_ORDER_OFFS[next_u],
            C_ORDER_OFFS[next_u + 1],
            lane_id
        );
        if (found)
        {
            for (uint32_t j = 0; j < depth; j++)
            {
                if (partial_result[warp_id][order[warp_id].u[j]] == min_array[elem_index])
                {
                    found = false;
                    break;
                }
            }
        }
        if (found)
        {
            auto group = cooperative_groups::coalesced_threads();
            auto rank = group.thread_rank();
            for (uint32_t j = 0; j < depth; j++)
            {
                new_results[(new_result_sizes_per_warp_prefix_sum[global_warp_id] + local_size[warp_id] + rank) * (depth + 1) + j] = partial_result[warp_id][order[warp_id].u[j]];
            }
            new_results[(new_result_sizes_per_warp_prefix_sum[global_warp_id] + local_size[warp_id] + rank) * (depth + 1) + depth] = min_array[elem_index];
            if (rank == 0)
            {
                local_size[warp_id] += group.size();
            }
        }
    }
}

void bfsOneLevel(
    uint32_t num_warps,
    uint32_t*& partial_results,
    uint32_t& partial_result_count,
    InitialOrder initial_order,
    uint8_t next_u,
    uint8_t depth,
    uint8_t min_off
) {
    uint32_t *new_results;
    uint32_t new_result_size;
    uint32_t *size_per_warp;
    uint32_t *size_per_warp_prefix_sum;

    cudaErrorCheck(cudaMalloc(&size_per_warp, sizeof(uint32_t) * (num_warps + 1)));
    cudaErrorCheck(cudaMalloc(&size_per_warp_prefix_sum, sizeof(uint32_t) * (num_warps + 1)));

    if (depth == 0)
    {
        joinBFSFirstCount<<<DIV_CEIL(num_warps, WARP_PER_BLOCK), BLOCK_DIM>>>(
            next_u, min_off, size_per_warp
        );
    }
    else
    {
        bfsRemainingCount<<<DIV_CEIL(num_warps, WARP_PER_BLOCK), BLOCK_DIM>>>(
            partial_results, partial_result_count, size_per_warp,
            initial_order, next_u, depth
        );
    }

    cudaErrorCheck(cudaDeviceSynchronize());

    // prefix sum on new_sizes
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, 
        size_per_warp, size_per_warp_prefix_sum,
        num_warps + 1);
    // Allocate temporary storage
    cudaErrorCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    // Run exclusive prefix sum
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, temp_storage_bytes, 
        size_per_warp, size_per_warp_prefix_sum,
        num_warps + 1);
    cudaErrorCheck(cudaFree(d_temp_storage));

    cudaErrorCheck(cudaMemcpy(
        &new_result_size, 
        size_per_warp_prefix_sum + num_warps,
        sizeof(uint32_t),
        cudaMemcpyDeviceToHost));

    std::cout << "level " << static_cast<uint32_t>(depth) << " partial result count: " << partial_result_count << '\n';

    cudaErrorCheck(cudaMalloc(&new_results, sizeof(uint32_t) * new_result_size * (depth + 1)));

    if (depth == 0)
    {
        joinBFSFirstWrite<<<DIV_CEIL(num_warps, WARP_PER_BLOCK), BLOCK_DIM>>>(
            next_u, min_off, size_per_warp_prefix_sum,
            new_results
        );
    }
    else
    {
        bfsRemainingWrite<<<DIV_CEIL(num_warps, WARP_PER_BLOCK), BLOCK_DIM>>>(
            partial_results, partial_result_count, size_per_warp_prefix_sum, new_results,
            initial_order, next_u, depth
        );
    }

    cudaErrorCheck(cudaDeviceSynchronize());

    std::swap(new_result_size, partial_result_count);
    std::swap(new_results, partial_results);

    for (uint32_t j = 0u; j < num_warps; j++)
    {
        std::cout << size_per_warp[j] << ' ';
    }
    std::cout << '\n';
    /*for (uint32_t j = 0u; j < partial_result_count; j++)
    {
        for (uint32_t k = 0; k < i + 1; k++)
            std::cout << partial_results[j * (i + 1) + k] << ' ';
    }
    std::cout << '\n';*/

    if (new_results) cudaErrorCheck(cudaFree(new_results));
    cudaErrorCheck(cudaFree(size_per_warp));
    cudaErrorCheck(cudaFree(size_per_warp_prefix_sum));
}
