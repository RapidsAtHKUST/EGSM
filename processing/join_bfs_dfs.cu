
#include <cooperative_groups.h>
#include "utils/config.h"
#include "utils/types.h"
#include "utils/cuda_helpers.h"
#include "utils/globals.h"
#include "utils/mem_pool.h"
#include "graph/graph.h"
#include "structures/hashed_tries.h"
#include "processing/plan.h"
#include "processing/common.h"
#include "processing/join_bfs_dfs.h"

__global__ void freeMemory(uint32_t *next_ptr)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        cudaErrorCheckInKernel(cudaFree(next_ptr));
    }
}

__global__ void joinDFSGroupKernelKernel(
    uint32_t *res,
    unsigned long long int res_size,
    PoolElem new_res,
    unsigned long long int max_new_res_size,
    unsigned long long int *new_res_size,
    InitialOrder initial_order,
    uint8_t next_u,
    uint8_t next_u_min_off,
    uint8_t max_depth,
    bool only_one_group,
    uint32_t *pending_count
) {
    if (only_one_group && *new_res_size >= max_new_res_size) return;

    __shared__ uint32_t result_queue[WARP_PER_BLOCK][MAX_VCOUNT][WARP_SIZE];
    __shared__ uint8_t queue_pos[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ uint8_t queue_size[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ uint32_t *intersection_input[WARP_PER_BLOCK][MAX_ECOUNT * 2];
    __shared__ uint32_t intersection_input_sizes[WARP_PER_BLOCK][MAX_ECOUNT * 2];
    __shared__ uint32_t start[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ int8_t depth[WARP_PER_BLOCK];
    __shared__ int8_t initial_depth[WARP_PER_BLOCK];
    __shared__ int8_t mask_index[WARP_PER_BLOCK];

    __shared__ uint32_t num_partial_results[WARP_PER_BLOCK][MAX_VCOUNT];
    //__shared__ uint32_t next_partial_result_progress[WARP_PER_BLOCK];
    //__shared__ uint32_t *next_ptr[WARP_PER_BLOCK];

    // dynamic ordering
    __shared__ InitialOrder order[WARP_PER_BLOCK];
    __shared__ uint16_t mapped_vs[WARP_PER_BLOCK];
    __shared__ uint8_t min_offs[WARP_PER_BLOCK][MAX_VCOUNT];

    __shared__ GraphUtils s_utils;
    __shared__ uint32_t tries_vsizes[MAX_ECOUNT * 2];

    copyUtilsConstantToShared(s_utils);
    // move data to shared memory
    for (uint32_t i = threadIdx.x; i < C_NUM_EQ * 2; i += blockDim.x)
    {
        tries_vsizes[i] = tries.compacted_vs_sizes_[i];
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    unsigned long long int global_warp_id = blockIdx.x * WARP_PER_BLOCK + warp_id;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t cur_v;

    if (global_warp_id >= res_size)
    {
        return;
    }

    // initialize shared data
    for (uint32_t i = lane_id; i < MAX_ECOUNT * 2; i += WARP_SIZE)
    {
        intersection_input[warp_id][i] = NULL;
        intersection_input_sizes[warp_id][i] = UINT32_MAX;
    }
    __syncwarp();

    // build order, mapped_vs, depth
    {
        if (lane_id < MAX_VCOUNT)
        {
            order[warp_id].u[lane_id] = initial_order.u[lane_id];
        }
        __syncwarp();
        uint16_t bitmap = 0u;
        if (lane_id < C_NUM_VQ)
        {
            if (order[warp_id].u[lane_id] != UINT8_MAX)
            {
                bitmap = (1 << order[warp_id].u[lane_id]);
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
            mapped_vs[warp_id] = bitmap;
            initial_depth[warp_id] = __popc(mapped_vs[warp_id]);
            depth[warp_id] = __popc(mapped_vs[warp_id]);
            mask_index[warp_id] = 0;
        }
        __syncwarp();
    }
    __syncwarp();
    
    if (only_one_group && *new_res_size >= max_new_res_size) return;

    // fill in the initial partial result
    if (lane_id < depth[warp_id])
    {
        result_queue[warp_id][order[warp_id].u[lane_id]][order[warp_id].u[lane_id]] = 
            res[global_warp_id * depth[warp_id] + lane_id];
    }
    __syncwarp();
    if (lane_id < C_NUM_VQ)
    {
        queue_pos[warp_id][lane_id] = 
            ((1 << lane_id) & mapped_vs[warp_id]) ?
            lane_id : 0u;
        queue_size[warp_id][lane_id] = 
            ((1 << lane_id) & mapped_vs[warp_id]) ?
            lane_id + 1 : 0u;
        start[warp_id][lane_id] = 0u;
        min_offs[warp_id][lane_id] = UINT8_MAX;
        num_partial_results[warp_id][lane_id] = 0u;
    }
    __syncwarp();

    // set the intersection_input and intersection_input_sizes
    for (uint8_t i = 0u; i < depth[warp_id]; i++)
    {
        map_new_v(
            order[warp_id].u[i],
            result_queue[warp_id][order[warp_id].u[i]][queue_pos[warp_id][order[warp_id].u[i]]],
            mapped_vs[warp_id],
            intersection_input[warp_id],
            intersection_input_sizes[warp_id],
            s_utils,
            warp_id,
            lane_id
        );
    }
    __syncwarp();

    bool exit_warp = false;

    // enumeration
    while (depth[warp_id] >= initial_depth[warp_id])
    {
        __syncwarp();
        if (only_one_group && *new_res_size >= max_new_res_size) return;
        if (order[warp_id].u[depth[warp_id]] == UINT8_MAX)
        {
            // map a new query vertex, determine arrays to be intersect.
            if (depth[warp_id] == initial_depth[warp_id])
            {
                if (exit_warp) break;
#ifdef DEBUG
                if (lane_id == 0) printf("warp_id=%d, new level depth=%d\n", warp_id, depth[warp_id]);
#endif
                if (next_u != UINT8_MAX)
                {
                    if (lane_id == 0)
                    {
                        order[warp_id].u[depth[warp_id]] = next_u;
                        min_offs[warp_id][order[warp_id].u[depth[warp_id]]] = next_u_min_off;
                        mapped_vs[warp_id] |= (1 << order[warp_id].u[depth[warp_id]]);
                        start[warp_id][order[warp_id].u[depth[warp_id]]] = global_warp_id * WARP_SIZE;

                        if (intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] == UINT32_MAX)
                        {
                            intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] = 
                                min(tries_vsizes[min_offs[warp_id][order[warp_id].u[depth[warp_id]]]],
                                start[warp_id][order[warp_id].u[depth[warp_id]]] + WARP_SIZE);
                        }
                        else
                        {
                            intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] = 
                                min(intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]],
                                start[warp_id][order[warp_id].u[depth[warp_id]]] + WARP_SIZE);
                        }
                    }
                    __syncwarp();
                    if (start[warp_id][order[warp_id].u[depth[warp_id]]] >= intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]])
                    {
                        break;
                    }
                }
                else
                {
                    set_next_u(
                        mapped_vs[warp_id], start[warp_id],
                        intersection_input_sizes[warp_id], tries_vsizes,
                        order[warp_id].u[depth[warp_id]], min_offs[warp_id],
                        mask_index[warp_id],
                        s_utils, warp_id, lane_id
                    );

                    __syncwarp();
                }
                if (next_u != UINT8_MAX) exit_warp = true;
            }
            else
            {
#ifdef DEBUG
                if (lane_id == 0) printf("warp_id=%d, new level depth=%d\n", warp_id, depth[warp_id]);
                __syncwarp();
#endif
                set_next_u(
                    mapped_vs[warp_id],
                    start[warp_id],
                    intersection_input_sizes[warp_id],
                    tries_vsizes,
                    order[warp_id].u[depth[warp_id]],
                    min_offs[warp_id],
                    mask_index[warp_id],
                    s_utils,
                    warp_id,
                    lane_id
                );
                __syncwarp();
            }
        }
        __syncwarp();

        if (only_one_group && *new_res_size >= max_new_res_size) return;
        __syncwarp();

        if (queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] >= queue_size[warp_id][order[warp_id].u[depth[warp_id]]])
        {
            if (start[warp_id][order[warp_id].u[depth[warp_id]]] >= intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]])
            {
                // no more intersection to do, backtrack
                if (lane_id == 0)
                {
#ifdef DEBUG
                    printf("warp_id=%d, backtrack depth %d->%d\n", warp_id, depth[warp_id], depth[warp_id] - 1);
#endif
                    queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                    queue_size[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                    start[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                }
                __syncwarp();
                unmap_u(
                    order[warp_id].u[depth[warp_id]],
                    mapped_vs[warp_id], intersection_input[warp_id], intersection_input_sizes[warp_id],
                    s_utils, warp_id, lane_id, tries
                );
                if (lane_id == 0)
                {
                    if (intersection_input[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] == NULL)
                    {
                        intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] = UINT32_MAX;
                    }
                    num_partial_results[warp_id][depth[warp_id] - 1] += num_partial_results[warp_id][depth[warp_id]];

                    num_partial_results[warp_id][depth[warp_id]] = 0u;

                    mapped_vs[warp_id] &= (~(1 << order[warp_id].u[depth[warp_id]]));
                    order[warp_id].u[depth[warp_id]] = UINT8_MAX;

                    while (((~mapped_vs[warp_id]) & c_plan.masks_[mask_index[warp_id] - 1]) && mask_index[warp_id] > 0)
                    {
                        mask_index[warp_id] --;
                        //printf("unset mask %d\n", mask_index[warp_id]);
                    }
                    depth[warp_id]--;
                }
                __syncwarp();

                if (depth[warp_id] >= initial_depth[warp_id])
                {
                    if (lane_id == 0)
                    {
                        queue_pos[warp_id][order[warp_id].u[depth[warp_id]]]++;
                        //printf("depth=%d, order=%d\n", depth[warp_id], order[warp_id].u[depth[warp_id]]);
                        if (queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] >= queue_size[warp_id][order[warp_id].u[depth[warp_id]]])
                        {
                            cur_v = UINT32_MAX;
                        }
                        else
                        {
                            // map a new v to u
                            cur_v = result_queue[warp_id][order[warp_id].u[depth[warp_id]]][queue_pos[warp_id][order[warp_id].u[depth[warp_id]]]];
                        }
                    }
                    __syncwarp();
                    if (only_one_group && *new_res_size >= max_new_res_size) return;
                    cur_v = __shfl_sync(0xffffffff, cur_v, 0);
#ifdef DEBUG
                    if (lane_id == 0) printf("select vertex %d, and update its neighbors\n", cur_v);
                    __syncwarp();
#endif
                    if (cur_v == UINT32_MAX) continue;
                    //if (depth[warp_id] - initial_depth[warp_id] <= 1 || num_partial_results[warp_id][depth[warp_id]] < 100000)
                    {
                        map_new_v(
                            order[warp_id].u[depth[warp_id]],
                            cur_v,
                            mapped_vs[warp_id], intersection_input[warp_id],
                            intersection_input_sizes[warp_id],
                            s_utils,
                            warp_id,
                            lane_id
                        );
                    }
                }
            }
            else
            {
#ifdef DEBUG
                if (lane_id == 0) printf("warp_id=%d, start=%d, intersection_size=%d\n", warp_id, start[warp_id][order[warp_id].u[depth[warp_id]]], intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]]);
                __syncwarp();
#endif
                // need to start/continue the intersection
                bool found;
                if (lane_id == 0)
                {
                    queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                    queue_size[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                }
                __syncwarp();
                const uint32_t *min_array = 
                    intersection_input[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] == NULL ?
                    tries.compacted_vs_ + min_offs[warp_id][order[warp_id].u[depth[warp_id]]] * C_MAX_L_FREQ :
                    intersection_input[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]];
                
                const uint32_t& elem_index = start[warp_id][order[warp_id].u[depth[warp_id]]] + lane_id;

                if (only_one_group && *new_res_size >= max_new_res_size) return;

                found = gpuIntersection(
                    intersection_input[warp_id],
                    intersection_input_sizes[warp_id],
                    min_array,
                    min_offs[warp_id][order[warp_id].u[depth[warp_id]]],
                    start[warp_id][order[warp_id].u[depth[warp_id]]],
                    C_ORDER_OFFS[order[warp_id].u[depth[warp_id]]],
                    C_ORDER_OFFS[order[warp_id].u[depth[warp_id]] + 1],
                    lane_id
                );
                __syncwarp();
                // check if the vertex has already been visited
                if (found)
                {
                    for (uint8_t i = 0u; i < depth[warp_id]; i++)
                    {
                        if (result_queue[warp_id][order[warp_id].u[i]][queue_pos[warp_id][order[warp_id].u[i]]] == min_array[elem_index])
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (found)
                {
#ifdef DEBUG
                    printf("found %u\n", min_array[elem_index]);
#endif
                    auto group = cooperative_groups::coalesced_threads();
                    auto rank = group.thread_rank();
                    if (depth[warp_id] == max_depth - 1)
                    {
                        unsigned long long int write_pos = atomicAdd(new_res_size, 1ul);
                        if (only_one_group && write_pos < max_new_res_size)
                        {
#ifdef DEBUG
                            printf("num_results=%d\n", write_pos);
                            if (rank == 0) printf("complete results: \n");
                            group.sync();
                            for (uint32_t j = 0; j < C_NUM_VQ; j++)
                            {
                                if (j != order[warp_id].u[depth[warp_id]])
                                    printf("%d\t", result_queue[warp_id][j][queue_pos[warp_id][j]]);
                                else
                                    printf("%d\t", min_array[elem_index]);
                                if (rank == 0) printf("\n");
                                group.sync();
                            }
#endif
                            if (only_one_group)
                            {
                                for (uint32_t j = 0; j < max_depth; j++)
                                {
                                    if (j != depth[warp_id])
                                        C_MEMPOOL.array_[(new_res + write_pos * max_depth + c_plan.res_pos[order[warp_id].u[j]]) % C_MEMPOOL.capability_] = result_queue[warp_id][order[warp_id].u[j]][queue_pos[warp_id][order[warp_id].u[j]]];
                                    else
                                        C_MEMPOOL.array_[(new_res + write_pos * max_depth + c_plan.res_pos[order[warp_id].u[j]]) % C_MEMPOOL.capability_] = min_array[elem_index];
                                }
                            }
                            else
                            {
                                for (uint32_t j = 0; j < max_depth; j++)
                                {
                                    if (j != depth[warp_id])
                                        C_MEMPOOL.array_[(new_res + (write_pos % max_new_res_size) * max_depth + c_plan.res_pos[order[warp_id].u[j]]) % C_MEMPOOL.capability_] = result_queue[warp_id][order[warp_id].u[j]][queue_pos[warp_id][order[warp_id].u[j]]];
                                    else
                                        C_MEMPOOL.array_[(new_res + (write_pos % max_new_res_size) * max_depth + c_plan.res_pos[order[warp_id].u[j]]) % C_MEMPOOL.capability_] = min_array[elem_index];
                                }
                            }
                        }
                    }
                    else
                    {
                        result_queue[warp_id][order[warp_id].u[depth[warp_id]]][rank] = min_array[elem_index];

                        if (0 == rank) {
                            queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                            queue_size[warp_id][order[warp_id].u[depth[warp_id]]] = group.size();
                        }
                    }
                }
                __syncwarp();
                if (only_one_group && *new_res_size >= max_new_res_size) return;

                __syncwarp();
                if (lane_id == 0)
                {
                    num_partial_results[warp_id][depth[warp_id]] += 1;
                    start[warp_id][order[warp_id].u[depth[warp_id]]] = min(start[warp_id][order[warp_id].u[depth[warp_id]]] + WARP_SIZE,
                        intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]]);
#ifdef DEBUG
                    if (lane_id == 0) printf("warp_id=%d, start=%d, intersection_size=%d\n", warp_id, start[warp_id][order[warp_id].u[depth[warp_id]]], intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]]);
#endif
                }
                __syncwarp();
                if (depth[warp_id] < max_depth - 1 && queue_size[warp_id][order[warp_id].u[depth[warp_id]]] > 0)
                {
                    // update intersection_input and intersection_input_sizes for unmapped vertices
                    uint32_t cur_v = result_queue[warp_id][order[warp_id].u[depth[warp_id]]][0];
#ifdef DEBUG
                    if (lane_id == 0) printf("map %d to %d\n", order[warp_id].u[depth[warp_id]], cur_v);
                    __syncwarp();
#endif
                    map_new_v(
                        order[warp_id].u[depth[warp_id]], cur_v,
                        mapped_vs[warp_id], intersection_input[warp_id], intersection_input_sizes[warp_id],
                        s_utils, warp_id, lane_id
                    );
                    if (lane_id == 0)
                    {
                        depth[warp_id] ++;
#ifdef DEBUG
                        printf("next level depth: %d->%d\n", depth[warp_id] - 1, depth[warp_id]);
#endif
                    }
                }
            }
        }
        else
        {
            if (lane_id == 0)
            {
#ifdef DEBUG
                printf("move to queue_pos=%d, next candidate=%d, at depth=%d\n", queue_pos[warp_id][order[warp_id].u[depth[warp_id]]], result_queue[warp_id][order[warp_id].u[depth[warp_id]]][queue_pos[warp_id][order[warp_id].u[depth[warp_id]]]], depth[warp_id]);
#endif
                depth[warp_id] ++;
            }
        }
        __syncwarp();
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
) {
    if (only_one_group && *new_res_size >= max_new_res_size) return;

    __shared__ uint32_t result_queue[WARP_PER_BLOCK][MAX_VCOUNT][WARP_SIZE];
    __shared__ uint8_t queue_pos[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ uint8_t queue_size[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ uint32_t *intersection_input[WARP_PER_BLOCK][MAX_ECOUNT * 2];
    __shared__ uint32_t intersection_input_sizes[WARP_PER_BLOCK][MAX_ECOUNT * 2];
    __shared__ uint32_t start[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ int8_t depth[WARP_PER_BLOCK];
    __shared__ int8_t initial_depth[WARP_PER_BLOCK];
    __shared__ int8_t mask_index[WARP_PER_BLOCK];

    __shared__ uint32_t num_partial_results[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ unsigned long long int next_partial_result_progress[WARP_PER_BLOCK];
    __shared__ uint32_t *next_ptr[WARP_PER_BLOCK];

    // dynamic ordering
    __shared__ InitialOrder order[WARP_PER_BLOCK];
    __shared__ uint16_t mapped_vs[WARP_PER_BLOCK];
    __shared__ uint8_t min_offs[WARP_PER_BLOCK][MAX_VCOUNT];

    __shared__ GraphUtils s_utils;
    __shared__ uint32_t tries_vsizes[MAX_ECOUNT * 2];

    copyUtilsConstantToShared(s_utils);
    // move data to shared memory
    for (uint32_t i = threadIdx.x; i < C_NUM_EQ * 2; i += blockDim.x)
    {
        tries_vsizes[i] = tries.compacted_vs_sizes_[i];
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    unsigned long long int global_warp_id = blockIdx.x * WARP_PER_BLOCK + warp_id;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t cur_v;

    if (global_warp_id >= res_size)
    {
        return;
    }

    // initialize shared data
    for (uint32_t i = lane_id; i < MAX_ECOUNT * 2; i += WARP_SIZE)
    {
        intersection_input[warp_id][i] = NULL;
        intersection_input_sizes[warp_id][i] = UINT32_MAX;
    }
    __syncwarp();

    // build order, mapped_vs, depth
    {
        if (lane_id < MAX_VCOUNT)
        {
            order[warp_id].u[lane_id] = initial_order.u[lane_id];
        }
        __syncwarp();
        uint16_t bitmap = 0u;
        if (lane_id < C_NUM_VQ)
        {
            if (order[warp_id].u[lane_id] != UINT8_MAX)
            {
                bitmap = (1 << order[warp_id].u[lane_id]);
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
            mapped_vs[warp_id] = bitmap;
            initial_depth[warp_id] = __popc(mapped_vs[warp_id]);
            depth[warp_id] = __popc(mapped_vs[warp_id]);
            mask_index[warp_id] = 0;
        }
        __syncwarp();
    }
    
    if (only_one_group && *new_res_size >= max_new_res_size) return;
    // fill in the initial partial result
    if (lane_id < depth[warp_id])
    {
        result_queue[warp_id][order[warp_id].u[lane_id]][order[warp_id].u[lane_id]] = 
            C_MEMPOOL.array_[(res + global_warp_id * depth[warp_id] + lane_id) % C_MEMPOOL.capability_];
    }
    __syncwarp();
    if (lane_id < C_NUM_VQ)
    {
        queue_pos[warp_id][lane_id] = 
            ((1 << lane_id) & mapped_vs[warp_id]) ?
            lane_id : 0u;
        queue_size[warp_id][lane_id] = 
            ((1 << lane_id) & mapped_vs[warp_id]) ?
            lane_id + 1 : 0u;
        start[warp_id][lane_id] = 0u;
        min_offs[warp_id][lane_id] = UINT8_MAX;
        num_partial_results[warp_id][lane_id] = 0u;
    }
    __syncwarp();

    // set the intersection_input and intersection_input_sizes
    for (uint8_t i = 0u; i < depth[warp_id]; i++)
    {
        map_new_v(
            order[warp_id].u[i],
            result_queue[warp_id][order[warp_id].u[i]][queue_pos[warp_id][order[warp_id].u[i]]],
            mapped_vs[warp_id],
            intersection_input[warp_id], intersection_input_sizes[warp_id],
            s_utils,
            warp_id,
            lane_id);
    }
    __syncwarp();

    bool exit_warp = false;
    // enumeration
    while (depth[warp_id] >= initial_depth[warp_id])
    {
        __syncwarp();
        if (only_one_group && *new_res_size >= max_new_res_size) return;
        if (order[warp_id].u[depth[warp_id]] == UINT8_MAX)
        {
            // map a new query vertex, determine arrays to be intersect.
            if (depth[warp_id] == initial_depth[warp_id])
            {
                if (exit_warp) break;
#ifdef DEBUG
                if (lane_id == 0) printf("warp_id=%d, new level depth=%d\n", warp_id, depth[warp_id]);
#endif
                if (next_u != UINT8_MAX)
                {
                    if (lane_id == 0)
                    {
                        order[warp_id].u[depth[warp_id]] = next_u;
                        min_offs[warp_id][order[warp_id].u[depth[warp_id]]] = next_u_min_off;
                        mapped_vs[warp_id] |= (1 << order[warp_id].u[depth[warp_id]]);
                        start[warp_id][order[warp_id].u[depth[warp_id]]] = global_warp_id * WARP_SIZE;

                        if (intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] == UINT32_MAX)
                        {
                            intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] = 
                                min(tries_vsizes[min_offs[warp_id][order[warp_id].u[depth[warp_id]]]],
                                start[warp_id][order[warp_id].u[depth[warp_id]]] + WARP_SIZE);
                        }
                        else
                        {
                            intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] = 
                                min(intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]],
                                start[warp_id][order[warp_id].u[depth[warp_id]]] + WARP_SIZE);
                        }
                    }
                    __syncwarp();
                    if (start[warp_id][order[warp_id].u[depth[warp_id]]] >= intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]])
                    {
                        break;
                    }
                }
                else
                {
                    set_next_u(
                        mapped_vs[warp_id], start[warp_id],
                        intersection_input_sizes[warp_id], tries_vsizes,
                        order[warp_id].u[depth[warp_id]], min_offs[warp_id],
                        mask_index[warp_id],
                        s_utils, warp_id, lane_id
                    );

                    __syncwarp();
                }
                if (next_u != UINT8_MAX) exit_warp = true;
            }
            else
            {
#ifdef DEBUG
                if (lane_id == 0) printf("warp_id=%d, new level depth=%d\n", warp_id, depth[warp_id]);
                __syncwarp();
#endif
                set_next_u(
                    mapped_vs[warp_id],
                    start[warp_id],
                    intersection_input_sizes[warp_id],
                    tries_vsizes,
                    order[warp_id].u[depth[warp_id]],
                    min_offs[warp_id],
                    mask_index[warp_id],
                    s_utils,
                    warp_id,
                    lane_id
                );
                __syncwarp();
                
                if (C_LB_ENABLE && intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] > 1024)
                {
                    uint32_t num_warp = DIV_CEIL(intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]], 32);
                    cudaError_t cucheck_err = cudaSuccess;
                    if (lane_id == 0)
                    {
                        cucheck_err = cudaMalloc(&next_ptr[warp_id], sizeof(uint32_t) * depth[warp_id] * num_warp);
                    }
                    __shfl_sync(0xffffffff, cucheck_err, 0u);

                    if (cucheck_err == cudaSuccess)
                    {
                        for (uint32_t i = lane_id; i < num_warp; i += WARP_SIZE)
                        {
                            for (uint32_t j = 0; j < depth[warp_id]; j++)
                                next_ptr[warp_id][i * depth[warp_id] + j] = result_queue[warp_id][order[warp_id].u[j]][queue_pos[warp_id][order[warp_id].u[j]]];
                        }
                        __syncwarp();

                        if (lane_id == 0)
                        {
                            cudaStream_t s;
                            cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
                            uint8_t temp_order_u = order[warp_id].u[depth[warp_id]];
                            order[warp_id].u[depth[warp_id]] = UINT8_MAX;
                            atomicOr(lb_triggered, 1u);

                            joinDFSGroupKernelKernel<<<DIV_CEIL(num_warp, WARP_PER_BLOCK), BLOCK_DIM, 0, s>>>(
                                next_ptr[warp_id],
                                num_warp,
                                new_res,
                                max_new_res_size,
                                new_res_size,
                                order[warp_id], 
                                temp_order_u,
                                min_offs[warp_id][temp_order_u],
                                max_depth,
                                only_one_group,
                                pending_count
                            );
                            cudaErrorCheckInKernel(cudaGetLastError());
                            freeMemory<<<1, WARP_SIZE, 0, s>>>(next_ptr[warp_id]);
                            cudaErrorCheckInKernel(cudaGetLastError());
                            cudaStreamDestroy(s);

                            order[warp_id].u[depth[warp_id]] = temp_order_u;
                            queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] = queue_size[warp_id][order[warp_id].u[depth[warp_id]]];
                            start[warp_id][order[warp_id].u[depth[warp_id]]] = intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]];
                        }
                    }
                    else
                    {
                        if (lane_id == 0)
                        {
                            const char *err_str = cudaGetErrorString(cucheck_err);
                            printf("Skip load balancing due to %s at %s (%d)\n", err_str, __FILE__, __LINE__);
                        }
                    }
                    __syncwarp();
                }
            }
        }
        __syncwarp();

        if (only_one_group && *new_res_size >= max_new_res_size) return;
        __syncwarp();

        if (queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] >= queue_size[warp_id][order[warp_id].u[depth[warp_id]]])
        {
            if (start[warp_id][order[warp_id].u[depth[warp_id]]] >= intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]])
            {
                // no more intersection to do, backtrack
                if (lane_id == 0)
                {
#ifdef DEBUG
                    printf("warp_id=%d, backtrack depth %d->%d\n", warp_id, depth[warp_id], depth[warp_id] - 1);
#endif
                    queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                    queue_size[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                    start[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                }
                __syncwarp();
                unmap_u(
                    order[warp_id].u[depth[warp_id]],
                    mapped_vs[warp_id], intersection_input[warp_id], intersection_input_sizes[warp_id],
                    s_utils, warp_id, lane_id, tries
                );
                __syncwarp();
                if (lane_id == 0)
                {
                    if (intersection_input[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] == NULL)
                    {
                        intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] = UINT32_MAX;
                    }
                    num_partial_results[warp_id][depth[warp_id] - 1] += num_partial_results[warp_id][depth[warp_id]];

                    num_partial_results[warp_id][depth[warp_id]] = 0u;

                    mapped_vs[warp_id] &= (~(1 << order[warp_id].u[depth[warp_id]]));
                    order[warp_id].u[depth[warp_id]] = UINT8_MAX;

                    while (((~mapped_vs[warp_id]) & c_plan.masks_[mask_index[warp_id] - 1]) && mask_index[warp_id] > 0)
                    {
                        mask_index[warp_id] --;
                        //printf("unset mask %d\n", mask_index[warp_id]);
                    }
                    depth[warp_id]--;
                }
                __syncwarp();

                if (depth[warp_id] >= initial_depth[warp_id])
                {
                    if (lane_id == 0)
                    {
                        queue_pos[warp_id][order[warp_id].u[depth[warp_id]]]++;
                        //printf("depth=%d, order=%d\n", depth[warp_id], order[warp_id].u[depth[warp_id]]);
                        if (queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] >= queue_size[warp_id][order[warp_id].u[depth[warp_id]]])
                        {
                            cur_v = UINT32_MAX;
                        }
                        else
                        {
                            // map a new v to u
                            cur_v = result_queue[warp_id][order[warp_id].u[depth[warp_id]]][queue_pos[warp_id][order[warp_id].u[depth[warp_id]]]];
                        }
                    }
                    __syncwarp();
                    if (only_one_group && *new_res_size >= max_new_res_size) return;
                    cur_v = __shfl_sync(0xffffffff, cur_v, 0);
#ifdef DEBUG
                    if (lane_id == 0) printf("select vertex %d, and update its neighbors\n", cur_v);
                    __syncwarp();
#endif
                    if (cur_v == UINT32_MAX) continue;
                    if ((!C_LB_ENABLE) || depth[warp_id] - initial_depth[warp_id] <= 1 || num_partial_results[warp_id][depth[warp_id]] < 1000000)
                    {
                        map_new_v(
                            order[warp_id].u[depth[warp_id]], cur_v,
                            mapped_vs[warp_id], intersection_input[warp_id], intersection_input_sizes[warp_id],
                            s_utils, warp_id, lane_id
                        );
                    }
                    else
                    {
                        cudaError_t cucheck_err = cudaSuccess;
                        if (lane_id == 0)
                        {
                            //if (lane_id == 0) printf("xxtrigger %lu %d\n", global_warp_id, depth[warp_id]);
                            uint32_t max_size = queue_size[warp_id][order[warp_id].u[depth[warp_id]]] - queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] + intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] - start[warp_id][order[warp_id].u[depth[warp_id]]];
                            cucheck_err = cudaMalloc(&(next_ptr[warp_id]), sizeof(uint32_t) * (depth[warp_id] + 1) * max_size);
                        }
                        __shfl_sync(0xffffffff, cucheck_err, 0u);

                        if (cucheck_err == cudaSuccess)
                        {
                            // do other intersection
                            if (lane_id + queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] < queue_size[warp_id][order[warp_id].u[depth[warp_id]]])
                            {
                                for (uint8_t j = 0; j < depth[warp_id]; j++)
                                {
                                    next_ptr[warp_id][lane_id * (depth[warp_id] + 1) + j] = result_queue[warp_id][order[warp_id].u[j]][queue_pos[warp_id][order[warp_id].u[j]]];
                                }
                                next_ptr[warp_id][lane_id * (depth[warp_id] + 1) + depth[warp_id]] = result_queue[warp_id][order[warp_id].u[depth[warp_id]]][queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] + lane_id];
                            }
                            __syncwarp();
                            next_partial_result_progress[warp_id] = queue_size[warp_id][order[warp_id].u[depth[warp_id]]] - queue_pos[warp_id][order[warp_id].u[depth[warp_id]]];

                            while (start[warp_id][order[warp_id].u[depth[warp_id]]] < intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]])
                            {
                                bool found = false;
                                __syncwarp();
                                const uint32_t *min_array = 
                                    intersection_input[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] == NULL ?
                                    tries.compacted_vs_ + min_offs[warp_id][order[warp_id].u[depth[warp_id]]] * C_MAX_L_FREQ :
                                    intersection_input[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]];
                                
                                const uint32_t& elem_index = start[warp_id][order[warp_id].u[depth[warp_id]]] + lane_id;

                                found = gpuIntersection(
                                    intersection_input[warp_id],
                                    intersection_input_sizes[warp_id],
                                    min_array,
                                    min_offs[warp_id][order[warp_id].u[depth[warp_id]]],
                                    start[warp_id][order[warp_id].u[depth[warp_id]]],
                                    C_ORDER_OFFS[order[warp_id].u[depth[warp_id]]],
                                    C_ORDER_OFFS[order[warp_id].u[depth[warp_id]] + 1],
                                    lane_id
                                );
                                __syncwarp();
                                if (found)
                                {
                                    for (uint8_t i = 0u; i < depth[warp_id]; i++)
                                    {
                                        if (result_queue[warp_id][order[warp_id].u[i]][queue_pos[warp_id][order[warp_id].u[i]]] == min_array[elem_index])
                                        {
                                            found = false;
                                            break;
                                        }
                                    }
                                }
                                __syncwarp();
                                if (found)
                                {
                                    auto group = cooperative_groups::coalesced_threads();
                                    auto rank = group.thread_rank();
                                    for (uint8_t j = 0; j < depth[warp_id]; j++)
                                    {
                                        next_ptr[warp_id][(next_partial_result_progress[warp_id] + rank) * (depth[warp_id] + 1) + j] = result_queue[warp_id][order[warp_id].u[j]][queue_pos[warp_id][order[warp_id].u[j]]];
                                    }
                                    next_ptr[warp_id][(next_partial_result_progress[warp_id] + rank) * (depth[warp_id] + 1) + depth[warp_id]] = min_array[elem_index];
                                    if (rank == 0)
                                    {
                                        next_partial_result_progress[warp_id] += group.size();
                                    }
                                }
                                __syncwarp();
                                if (lane_id == 0)
                                {
                                    start[warp_id][order[warp_id].u[depth[warp_id]]] = min(
                                        start[warp_id][order[warp_id].u[depth[warp_id]]] + WARP_SIZE,
                                        intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]]
                                    );
                                }
                                __syncwarp();
                            }
                            __syncwarp();
                            if (lane_id == 0)
                            {
                                cudaStream_t s;
                                cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
                                atomicOr(lb_triggered, 1u);

                                joinDFSGroupKernelKernel<<<DIV_CEIL(next_partial_result_progress[warp_id], WARP_PER_BLOCK), BLOCK_DIM, 0, s>>>(
                                    next_ptr[warp_id],
                                    next_partial_result_progress[warp_id],
                                    new_res,
                                    max_new_res_size,
                                    new_res_size,
                                    order[warp_id], 
                                    UINT8_MAX,
                                    UINT8_MAX,
                                    max_depth,
                                    only_one_group,
                                    pending_count
                                );
                                cudaErrorCheckInKernel(cudaGetLastError());
                                freeMemory<<<1, WARP_SIZE, 0, s>>>(next_ptr[warp_id]);
                                cudaErrorCheckInKernel(cudaGetLastError());
                                cudaStreamDestroy(s);

                                queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] = queue_size[warp_id][order[warp_id].u[depth[warp_id]]];
                                start[warp_id][order[warp_id].u[depth[warp_id]]] = intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]];
                                num_partial_results[warp_id][depth[warp_id]] = 0u;
                            }
                        }
                        else
                        {
                            if (lane_id == 0)
                            {
                                const char *err_str = cudaGetErrorString(cucheck_err);
                                printf("Skip load balancing due to %s at %s (%d)\n", err_str, __FILE__, __LINE__);
                            }
                            __syncwarp();
                            map_new_v(
                                order[warp_id].u[depth[warp_id]], cur_v,
                                mapped_vs[warp_id], intersection_input[warp_id], intersection_input_sizes[warp_id],
                                s_utils, warp_id, lane_id
                            );
                        }
                        __syncwarp();
                    }
                    __syncwarp();
                }
            }
            else
            {
#ifdef DEBUG
                if (lane_id == 0) printf("warp_id=%d, start=%d, intersection_size=%d\n", warp_id, start[warp_id][order[warp_id].u[depth[warp_id]]], intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]]);
                __syncwarp();
#endif
                // need to start/continue the intersection
                bool found;
                if (lane_id == 0)
                {
                    queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                    queue_size[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                }
                __syncwarp();
                const uint32_t *min_array = 
                    intersection_input[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]] == NULL ?
                    tries.compacted_vs_ + min_offs[warp_id][order[warp_id].u[depth[warp_id]]] * C_MAX_L_FREQ :
                    intersection_input[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]];
                
                const uint32_t& elem_index = start[warp_id][order[warp_id].u[depth[warp_id]]] + lane_id;

                if (only_one_group && *new_res_size >= max_new_res_size) return;
                found = gpuIntersection(
                    intersection_input[warp_id],
                    intersection_input_sizes[warp_id],
                    min_array,
                    min_offs[warp_id][order[warp_id].u[depth[warp_id]]],
                    start[warp_id][order[warp_id].u[depth[warp_id]]],
                    C_ORDER_OFFS[order[warp_id].u[depth[warp_id]]],
                    C_ORDER_OFFS[order[warp_id].u[depth[warp_id]] + 1],
                    lane_id
                );
                __syncwarp();
                // check if the vertex has already been visited
                if (found)
                {
                    for (uint8_t i = 0u; i < depth[warp_id]; i++)
                    {
                        if (result_queue[warp_id][order[warp_id].u[i]][queue_pos[warp_id][order[warp_id].u[i]]] == min_array[elem_index])
                        {
                            found = false;
                            break;
                        }
                    }
                }
                __syncwarp();
                if (found)
                {
#ifdef DEBUG
                    printf("found %u\n", min_array[elem_index]);
#endif
                    auto group = cooperative_groups::coalesced_threads();
                    auto rank = group.thread_rank();
                    if (depth[warp_id] == max_depth - 1)
                    {
                        unsigned long long int write_pos = atomicAdd(new_res_size, 1ul);
                        if (only_one_group && write_pos < max_new_res_size)
                        {
#ifdef DEBUG
                            printf("num_results=%d\n", write_pos);
                            if (rank == 0) printf("complete results: \n");
                            group.sync();
                            for (uint32_t j = 0; j < C_NUM_VQ; j++)
                            {
                                if (j != order[warp_id].u[depth[warp_id]])
                                    printf("%d\t", result_queue[warp_id][j][queue_pos[warp_id][j]]);
                                else
                                    printf("%d\t", min_array[elem_index]);
                                if (rank == 0) printf("\n");
                                group.sync();
                            }
#endif
                            if (only_one_group)
                            {
                                for (uint32_t j = 0; j < max_depth; j++)
                                {
                                    if (j != depth[warp_id])
                                        C_MEMPOOL.array_[(new_res + write_pos * max_depth + c_plan.res_pos[order[warp_id].u[j]]) % C_MEMPOOL.capability_] = result_queue[warp_id][order[warp_id].u[j]][queue_pos[warp_id][order[warp_id].u[j]]];
                                    else
                                        C_MEMPOOL.array_[(new_res + write_pos * max_depth + c_plan.res_pos[order[warp_id].u[j]]) % C_MEMPOOL.capability_] = min_array[elem_index];
                                }
                            }
                            else
                            {
                                for (uint32_t j = 0; j < max_depth; j++)
                                {
                                    if (j != depth[warp_id])
                                        C_MEMPOOL.array_[(new_res + (write_pos % max_new_res_size) * max_depth + c_plan.res_pos[order[warp_id].u[j]]) % C_MEMPOOL.capability_] = result_queue[warp_id][order[warp_id].u[j]][queue_pos[warp_id][order[warp_id].u[j]]];
                                    else
                                        C_MEMPOOL.array_[(new_res + (write_pos % max_new_res_size) * max_depth + c_plan.res_pos[order[warp_id].u[j]]) % C_MEMPOOL.capability_] = min_array[elem_index];
                                }
                            }
                        }
                    }
                    else
                    {
                        result_queue[warp_id][order[warp_id].u[depth[warp_id]]][rank] = min_array[elem_index];

                        if (0 == rank) {
                            queue_pos[warp_id][order[warp_id].u[depth[warp_id]]] = 0u;
                            queue_size[warp_id][order[warp_id].u[depth[warp_id]]] = group.size();
                        }
                    }
                }
                __syncwarp();
                if (only_one_group && *new_res_size >= max_new_res_size) return;

                __syncwarp();
                if (lane_id == 0)
                {
                    num_partial_results[warp_id][depth[warp_id]] += 1;
                    start[warp_id][order[warp_id].u[depth[warp_id]]] = min(start[warp_id][order[warp_id].u[depth[warp_id]]] + WARP_SIZE,
                        intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]]);
#ifdef DEBUG
                    if (lane_id == 0) printf("warp_id=%d, start=%d, intersection_size=%d\n", warp_id, start[warp_id][order[warp_id].u[depth[warp_id]]], intersection_input_sizes[warp_id][min_offs[warp_id][order[warp_id].u[depth[warp_id]]]]);
#endif
                }
                __syncwarp();
                if (depth[warp_id] < max_depth - 1 && queue_size[warp_id][order[warp_id].u[depth[warp_id]]] > 0)
                {
                    // update intersection_input and intersection_input_sizes for unmapped vertices
                    uint32_t cur_v = result_queue[warp_id][order[warp_id].u[depth[warp_id]]][0];
#ifdef DEBUG
                    if (lane_id == 0) printf("map %d to %d\n", order[warp_id].u[depth[warp_id]], cur_v);
                    __syncwarp();
#endif
                    map_new_v(
                        order[warp_id].u[depth[warp_id]],
                        cur_v,
                        mapped_vs[warp_id],
                        intersection_input[warp_id],
                        intersection_input_sizes[warp_id],
                        s_utils,
                        warp_id,
                        lane_id
                    );
                    if (lane_id == 0)
                    {
                        depth[warp_id] ++;
#ifdef DEBUG
                        printf("next level depth: %d->%d\n", depth[warp_id] - 1, depth[warp_id]);
#endif
                    }
                }
            }
        }
        else
        {
            if (lane_id == 0)
            {
#ifdef DEBUG
                printf("move to queue_pos=%d, next candidate=%d, at depth=%d\n", queue_pos[warp_id][order[warp_id].u[depth[warp_id]]], result_queue[warp_id][order[warp_id].u[depth[warp_id]]][queue_pos[warp_id][order[warp_id].u[depth[warp_id]]]], depth[warp_id]);
#endif
                depth[warp_id] ++;
            }
        }
        __syncwarp();
    }
}
