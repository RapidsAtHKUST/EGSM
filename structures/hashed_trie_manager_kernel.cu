#include <cooperative_groups.h>

#include "utils/globals.h"
#include "utils/search.cuh"
#include "graph/graph_gpu.h"
#include "structures/hashed_tries.h"
#include "structures/hashed_trie_manager.h"


__global__ void compareNLF(
    const GraphGPU query,
    const GraphGPU data,
    const uint32_t *query_nlf,
    uint32_t *progress,
    uint32_t *flags,
    const uint32_t num_flags,
    uint32_t *candidate_buffer,
    uint32_t *num_candidates
) {
    __shared__ uint32_t local_nlf[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ uint32_t s_query_nlf[MAX_VCOUNT * MAX_VCOUNT];
    __shared__ uint32_t v_start[WARP_PER_BLOCK];
    __shared__ uint32_t local_flags[WARP_PER_BLOCK][MAX_VCOUNT];
    __shared__ uint32_t local_num_candidates[WARP_PER_BLOCK][MAX_VCOUNT];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t off1, off2, off, v, v_other, v_other_label;

    for (uint32_t i = threadIdx.x; i < C_NUM_VQ * C_NUM_LQ; i += blockDim.x)
    {
        s_query_nlf[i] = query_nlf[i];
    }
    __syncthreads();

    while (true)
    {
        if (lane_id == 0)
        {
            v_start[warp_id] = atomicAdd(progress, WARP_SIZE);
        }
        __syncwarp();
        if (v_start[warp_id] >= C_NUM_VD) break;

        if (lane_id < C_NUM_VQ)
        {
            local_flags[warp_id][lane_id] = UINT32_MAX;
            if (v_start[warp_id] + WARP_SIZE > C_NUM_VD)
            {
                local_flags[warp_id][lane_id] >>= (v_start[warp_id] + WARP_SIZE - C_NUM_VD);
            }
            local_num_candidates[warp_id][lane_id] = 0u;
        }
        __syncwarp();

        for (uint32_t i = 0u; i < min(C_NUM_VD - v_start[warp_id], WARP_SIZE); i++)
        {
            v = v_start[warp_id] + i;

            if (lane_id < C_NUM_LQ)
            {
                local_nlf[warp_id][lane_id] = 0u;
            }
            __syncwarp();

            // a warp traverses all the neighbors of v to compute the nlf
            off1 = data.offsets_[v], off2 = data.offsets_[v + 1];
            for (off = lane_id + off1; off < off2; off += WARP_SIZE)
            {
                v_other = data.neighbors_[off];
                v_other_label = data.vlabels_[v_other];

                if (v_other_label >= C_NUM_LQ) continue;

                atomicAdd(local_nlf[warp_id] + v_other_label, 1u);
            }
            __syncwarp();

            // a warp compare the nlf of v with that of each query vertex
            for (uint32_t j = 0; j < C_NUM_VQ; j++)
            {
                bool not_pass = lane_id < C_NUM_LQ &&
                    (data.vlabels_[v] != query.vlabels_[j] ||
                    local_nlf[warp_id][lane_id] < s_query_nlf[j * C_NUM_LQ + lane_id]);
                
                if (__ballot_sync(0xffffffff, not_pass) > 0 && lane_id == 0)
                {
                    local_flags[warp_id][j] &= ~(1 << (i % 32));
                }
            }
            __syncwarp();
        }
        // when a warp finish processing 32 vertices, write shared memory to global memory
        if (lane_id < C_NUM_VQ)
        {
            flags[lane_id * num_flags + v_start[warp_id] / 32] = local_flags[warp_id][lane_id];
            local_num_candidates[warp_id][lane_id] = atomicAdd(num_candidates + lane_id, __popc(local_flags[warp_id][lane_id]));
        }
        __syncwarp();
        for (uint32_t i = 0; i < C_NUM_VQ; i++)
        {
            if ((local_flags[warp_id][i] & (1 << lane_id)) && (v_start[warp_id] + lane_id < C_NUM_VD))
            {
                auto group = cooperative_groups::coalesced_threads();
                auto rank = group.thread_rank();
                candidate_buffer[i * C_MAX_L_FREQ + local_num_candidates[warp_id][i] + rank] = v_start[warp_id] + lane_id;
            }
        }
    }
}

__global__ void buildHashKeys(
    const uint32_t *in,
    const uint32_t in_size,
    uint32_t *keys0,
    uint32_t *keys1,
    const uint32_t num_bucket,
    const uint32_t C0,
    const uint32_t C1,
    const uint32_t C2,
    const uint32_t C3,
    uint32_t *progress,
    uint32_t *success
) {
    // a warp inserts at most 4 keys to the hash table at a time
    __shared__ uint32_t v[WARP_PER_BLOCK][4];
    __shared__ uint32_t bucket_index[WARP_PER_BLOCK][4];
    __shared__ uint32_t table_index[WARP_PER_BLOCK][4];

    __shared__ uint32_t idx_start[WARP_PER_BLOCK];
    __shared__ uint32_t nloops[WARP_PER_BLOCK];

    __shared__ uint32_t *keys[2];
    __shared__ uint32_t C[2][2];

    if (threadIdx.x == 0)
    {
        keys[0] = keys0;
        keys[1] = keys1;
        C[0][0] = C0;
        C[0][1] = C1;
        C[1][0] = C2;
        C[1][1] = C3;
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t mask = 0xff << (lane_id / 8) * 8;

    uint32_t pre_value, result, leader, elem_idx = lane_id / 8;

    while (true)
    {
        if (lane_id == 0)
        {
            idx_start[warp_id] = atomicAdd(progress, 4u);
        }
        __syncwarp();
        if (idx_start[warp_id] > in_size || *success != 0u) break;

        if (lane_id < 4)
        {
            v[warp_id][lane_id] = idx_start[warp_id] + lane_id < in_size ?
                in[idx_start[warp_id] + lane_id] :
                UINT32_MAX;
            table_index[warp_id][lane_id] = 0u;
        }
        if (lane_id == 0) nloops[warp_id] = 0u;
        __syncwarp();

        while (nloops[warp_id] < MAX_CUCKOO_LOOP)
        {
            if (*success != 0u) break;

            result = __ballot_sync(0xffffffff, v[warp_id][elem_idx] != UINT32_MAX);
            if (result == 0u) break;

            if (lane_id < 4 && v[warp_id][lane_id] != UINT32_MAX)
            {
                bucket_index[warp_id][lane_id] = 
                    (C[table_index[warp_id][lane_id]][0] ^ v[warp_id][lane_id] + C[table_index[warp_id][lane_id]][1]) % num_bucket;
            }
            __syncwarp();

            if (v[warp_id][elem_idx] != UINT32_MAX)
            {
                pre_value = keys[table_index[warp_id][elem_idx]][bucket_index[warp_id][elem_idx] * 8 + lane_id % 8];

                result = __ballot_sync(mask, pre_value == UINT32_MAX);

                while (result != 0)
                {
                    leader = __ffs(result) - 1;
                    if (lane_id == leader)
                    {
                        if (atomicCAS(keys[table_index[warp_id][elem_idx]] + bucket_index[warp_id][elem_idx] * 8 + lane_id % 8, pre_value, v[warp_id][elem_idx]) == pre_value)
                        {
                            v[warp_id][elem_idx] = UINT32_MAX;
                        }
                    }
                    __syncwarp(mask);
                    if (v[warp_id][elem_idx] == UINT32_MAX)
                    {
                        break;
                    }
                    pre_value = keys[table_index[warp_id][elem_idx]][bucket_index[warp_id][elem_idx] * 8 + lane_id % 8];
                    result = __ballot_sync(mask, pre_value == UINT32_MAX);
                }
                if (v[warp_id][elem_idx] != UINT32_MAX)
                {
                    leader = v[warp_id][elem_idx] % BUCKET_DIM;
                    if (lane_id % BUCKET_DIM == leader)
                    {
                        pre_value = atomicExch(keys[table_index[warp_id][elem_idx]] + bucket_index[warp_id][elem_idx] * 8 + lane_id % 8, v[warp_id][elem_idx]);
                        v[warp_id][elem_idx] = pre_value;
                        table_index[warp_id][elem_idx] = (table_index[warp_id][elem_idx] + 1) % 2;
                    }
                    __syncwarp(mask);
                }
            }
            __syncwarp();
            if (lane_id == 0) nloops[warp_id]++;
            __syncwarp();
        }
        if (*success != 0u)
        {
            break;
        }
        if (nloops[warp_id] >= MAX_CUCKOO_LOOP)
        {
            atomicAdd(success, 1u);
            break;
        }
    }
}


__global__ void buildHashValuesCount(
    const GraphGPU data,
    const uint32_t *flags_second_level,
    const uint32_t num_flags,
    uint32_t *progress,
    uint32_t *hash_keys,
    uint32_t *hash_values,
    const uint32_t num_bucket
) {
    __shared__ uint32_t v[WARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t idx[WARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t s_hash_values[WARP_PER_BLOCK][WARP_SIZE];

    __shared__ uint32_t table_start[WARP_PER_BLOCK];
    __shared__ uint8_t queue_size[WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t off1, off2, off, v_other;

    while (true)
    {
        if (lane_id == 0)
        {
            table_start[warp_id] = atomicAdd(progress, 32u);
        }
        __syncwarp();

        if (table_start[warp_id] >= num_bucket * BUCKET_DIM) break;

        if (lane_id == 0)
        {
            queue_size[warp_id] = 0u;
        }
        s_hash_values[warp_id][lane_id] = 0u;
        __syncwarp();

        // write vertices and their index to shared memory
        if (
            table_start[warp_id] + lane_id < num_bucket * BUCKET_DIM && 
            hash_keys[table_start[warp_id] + lane_id] != UINT32_MAX
        ) {
            auto group = cooperative_groups::coalesced_threads();
            auto rank = group.thread_rank();
            idx[warp_id][rank] = lane_id;
            v[warp_id][rank] = hash_keys[table_start[warp_id] + lane_id];
            if (rank == 0)
            {
                queue_size[warp_id] = group.size();
            }
        }
        __syncwarp();

        // process each vertex
        for (uint32_t i = 0; i < queue_size[warp_id]; i++)
        {
            off1 = data.offsets_[v[warp_id][i]];
            off2 = data.offsets_[v[warp_id][i] + 1];
            for (off = off1 + lane_id; off < off2; off += WARP_SIZE)
            {
                v_other = data.neighbors_[off];

                if ((flags_second_level[v_other / 32] & (1 << (v_other % 32))) > 0)
                {
                    atomicAdd(&s_hash_values[warp_id][idx[warp_id][i]], 1u);
                }
            }
            __syncwarp();
        }
        if (lane_id < queue_size[warp_id])
        {
            hash_values[(table_start[warp_id] + idx[warp_id][lane_id]) * 2] = s_hash_values[warp_id][idx[warp_id][lane_id]];
            // if a vertex does not have any neighbor, remove the vertex from the hash table
            if (s_hash_values[warp_id][idx[warp_id][lane_id]] == 0)
            {
                hash_keys[table_start[warp_id] + idx[warp_id][lane_id]] = UINT32_MAX;
            }
        }
        __syncwarp();
    }
}

__global__ void buildHashValuesWrite(
    const GraphGPU data,
    const uint32_t *flags_second_level,
    const uint32_t num_flags,
    uint32_t *progress,
    uint32_t *hash_keys,
    uint32_t *hash_values,
    const uint32_t num_bucket,
    uint32_t *neighbors
) {
    __shared__ uint32_t v[WARP_PER_BLOCK][WARP_SIZE];
    __shared__ uint32_t write_pos[WARP_PER_BLOCK][WARP_SIZE];

    __shared__ uint32_t table_start[WARP_PER_BLOCK];
    __shared__ uint8_t queue_size[WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t off1, off2, off, v_other;

    while (true)
    {
        if (lane_id == 0)
        {
            table_start[warp_id] = atomicAdd(progress, 32u);
        }
        __syncwarp();

        if (table_start[warp_id] >= num_bucket * BUCKET_DIM) break;

        if (lane_id == 0)
        {
            queue_size[warp_id] = 0u;
        }
        __syncwarp();

        // compute the number of neighbors of each vertex from hash_values
        if (table_start[warp_id] + lane_id < num_bucket * BUCKET_DIM)
        {
            hash_values[(table_start[warp_id] + lane_id) * 2 + 1] -= hash_values[(table_start[warp_id] + lane_id) * 2];
        }
        __syncwarp();
        
        // write vertices and their index to shared memory
        if (
            table_start[warp_id] + lane_id < num_bucket * BUCKET_DIM && 
            hash_keys[table_start[warp_id] + lane_id] != UINT32_MAX
        ) {
            auto group = cooperative_groups::coalesced_threads();
            auto rank = group.thread_rank();
            v[warp_id][rank] = hash_keys[table_start[warp_id] + lane_id];
            write_pos[warp_id][rank] = hash_values[(table_start[warp_id] + lane_id) * 2];
            if (rank == 0)
            {
                queue_size[warp_id] = group.size();
            }
        }
        __syncwarp();

        // process each vertex
        for (uint32_t i = 0; i < queue_size[warp_id]; i++)
        {
            off1 = data.offsets_[v[warp_id][i]];
            off2 = data.offsets_[v[warp_id][i] + 1];
            for (off = off1 + lane_id; off < off2; off += WARP_SIZE)
            {
                v_other = data.neighbors_[off];

                if ((flags_second_level[v_other / 32] & (1 << (v_other % 32))) > 0)
                {
                    auto group = cooperative_groups::coalesced_threads();
                    auto rank = group.thread_rank();
                    neighbors[write_pos[warp_id][i] + rank] = v_other;
                    if (rank == 0)
                    {
                        write_pos[warp_id][i] += group.size();
                    }
                }
                __syncwarp(__activemask());
            }
            __syncwarp();
        }
    }
}


__global__ void semiJoin(
    uint32_t* manager_key_flags0,
    uint32_t* manager_key_flags1,
    uint32_t* manager_neighbor_flags0,
    uint32_t* manager_neighbor_flags1,
    const uint32_t lidx,
    const uint32_t ridx,
    const uint32_t reversed_ridx,
    uint32_t *progress
) {
    __shared__ uint32_t deleted_keys[WARP_PER_BLOCK][WARP_SIZE * 2];
    __shared__ uint32_t deleted_gidx[WARP_PER_BLOCK][WARP_SIZE * 2];

    __shared__ uint32_t table_start[WARP_PER_BLOCK];
    __shared__ uint8_t queue_size[WARP_PER_BLOCK];
    __shared__ uint32_t nbr_start[WARP_PER_BLOCK];
    __shared__ uint32_t nbr_size[WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t idx, nbr;
    uint2 result;

    while (true)
    {
        if (lane_id == 0)
        {
            table_start[warp_id] = atomicAdd(progress, 32u);
        }
        __syncwarp();

        if (table_start[warp_id] >= tries.num_buckets_[ridx] * BUCKET_DIM) break;

        // the position within a hash table to check by a thread
        idx = table_start[warp_id] + lane_id;
        for (uint32_t i = 0; i < 2; i++)
        {
            if (lane_id == 0) queue_size[warp_id] = 0u;
            __syncwarp();

            // each thread check a key and modify the flag
            if (
                idx < tries.num_buckets_[ridx] * BUCKET_DIM &&
                tries.keys_[i][tries.BIdx(lidx, idx)] == UINT32_MAX &&
                tries.keys_[i][tries.BIdx(ridx, idx)] != UINT32_MAX
            ) {
                auto group = cooperative_groups::coalesced_threads();
                auto rank = group.thread_rank();

                deleted_keys[warp_id][rank] = tries.keys_[i][tries.BIdx(ridx, idx)];
                deleted_gidx[warp_id][rank] = tries.BIdx(ridx, idx);

                tries.keys_[i][tries.BIdx(ridx, idx)] = UINT32_MAX;
                if (rank == 0)
                {
                    queue_size[warp_id] = group.size();
                    //printf("size: %u\n", queue_size[warp_id]);
                }
            }
            __syncwarp();

            // process each key in the queue
            for (uint32_t j = 0; j < queue_size[warp_id]; j++)
            {
                if (lane_id == 0)
                {
                    nbr_start[warp_id] = tries.values_[i][deleted_gidx[warp_id][j] * 2];
                    nbr_size[warp_id] = tries.values_[i][deleted_gidx[warp_id][j] * 2 + 1];
                    //printf("nbr size: %u\n", nbr_size[warp_id]);
                }
                __syncwarp();

                // each thread check a neighbor of the key
                for (uint32_t k = lane_id; k < nbr_size[warp_id]; k += WARP_SIZE)
                {
                    nbr = tries.neighbors_[i][nbr_start[warp_id] + k];
                    //printf("remove (%u, %u)\n", deleted_keys[warp_id][j], nbr);
                    // remove the edge (deleted_keys[warp_id][j], nbr) from table[reversed_ridx]
                    result = tries.HashSearch(
                        reversed_ridx,
                        nbr
                    );
                    // remove deleted_keys from the neighbor of nbr in the reversed hash table
                    if (result.x != UINT32_MAX)
                    {
                        if (result.y == 0)
                        {
                            // binary search from the neighbor array
                            uint32_t pos = lower_bound(
                                tries.neighbors_[0] + tries.values_[0][(tries.hash_table_offs_[reversed_ridx] + result.x) * 2], 
                                tries.values_[0][(tries.hash_table_offs_[reversed_ridx] + result.x) * 2 + 1],
                                deleted_keys[warp_id][j]);
                            if (pos != UINT32_MAX && tries.neighbors_[0][tries.values_[0][(tries.hash_table_offs_[reversed_ridx] + result.x) * 2] + pos] == deleted_keys[warp_id][j])
                            {
                                manager_neighbor_flags0[tries.values_[0][(tries.hash_table_offs_[reversed_ridx] + result.x) * 2] + pos] = 1u;
                                atomicOr(manager_key_flags0 + tries.hash_table_offs_[reversed_ridx] + result.x, 1u);
                            }
                        }
                        else
                        {
                            // binary search from the neighbor array
                            uint32_t pos = lower_bound(
                                tries.neighbors_[1] + tries.values_[1][(tries.hash_table_offs_[reversed_ridx] + result.x) * 2], 
                                tries.values_[1][(tries.hash_table_offs_[reversed_ridx] + result.x) * 2 + 1],
                                deleted_keys[warp_id][j]);
                            if (pos != UINT32_MAX && tries.neighbors_[1][tries.values_[1][(tries.hash_table_offs_[reversed_ridx] + result.x) * 2] + pos] == deleted_keys[warp_id][j])
                            {
                                manager_neighbor_flags1[tries.values_[1][(tries.hash_table_offs_[reversed_ridx] + result.x) * 2] + pos] = 1u;
                                atomicOr(manager_key_flags1 + tries.hash_table_offs_[reversed_ridx] + result.x, 1u);
                            }
                        }
                    }
                    __syncwarp(__activemask());
                }
                __syncwarp();
                if (lane_id == 0)
                {
                    tries.values_[i][deleted_gidx[warp_id][j] * 2 + 1] = 0;
                }
                __syncwarp();
            }
        }
    }
}

__global__ void compactMiddleLevel(
    const HashedTrieManager manager,
    HashedTries tries,
    uint32_t *progress
) {
    __shared__ uint32_t modified_gidx[WARP_PER_BLOCK][WARP_SIZE * 2];

    __shared__ uint32_t table_start[WARP_PER_BLOCK];
    __shared__ uint8_t queue_size[WARP_PER_BLOCK];
    __shared__ uint32_t nbr_start[WARP_PER_BLOCK];
    __shared__ uint32_t nbr_size[WARP_PER_BLOCK];
    __shared__ uint32_t nbr_size_new[WARP_PER_BLOCK];

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint32_t idx;
    __shared__ uint32_t total_size;

    if (threadIdx.x == 0) total_size = tries.hash_table_offs_[C_NUM_EQ * 2];
    __syncthreads();

    while (true)
    {
        if (lane_id == 0)
        {
            table_start[warp_id] = atomicAdd(progress, 32u);
        }
        __syncwarp();

        if (table_start[warp_id] >= total_size) break;

        idx = table_start[warp_id] + lane_id;
        for (uint32_t i = 0; i < 2; i++)
        {
            if (lane_id == 0) queue_size[warp_id] = 0u;
            __syncwarp();

            // each thread check a key and modify the flag
            if (idx < total_size && tries.keys_[i][idx] != UINT32_MAX && manager.key_flags_[i][idx] == 1)
            {
                auto group = cooperative_groups::coalesced_threads();
                auto rank = group.thread_rank();

                modified_gidx[warp_id][rank] = idx;

                if (rank == 0)
                {
                    queue_size[warp_id] = group.size();
                    //printf("size: %u\n", queue_size[warp_id]);
                }
            }
            __syncwarp();

            // process each key in the queue
            for (uint32_t j = 0; j < queue_size[warp_id]; j++)
            {
                if (lane_id == 0)
                {
                    nbr_start[warp_id] = tries.values_[i][modified_gidx[warp_id][j] * 2];
                    nbr_size[warp_id] = tries.values_[i][modified_gidx[warp_id][j] * 2 + 1];
                    nbr_size_new[warp_id] = 0u;
                }
                __syncwarp();

                // each thread check a neighbor of the key
                for (uint32_t k = lane_id; k < nbr_size[warp_id]; k += WARP_SIZE)
                {
                    if (manager.neighbor_flags_[i][nbr_start[warp_id] + k] == 0u)
                    {
                        auto group = cooperative_groups::coalesced_threads();
                        auto rank = group.thread_rank();
                        uint32_t nbr = tries.neighbors_[i][nbr_start[warp_id] + k];
                        group.sync();
                        tries.neighbors_[i][nbr_start[warp_id] + nbr_size_new[warp_id] + rank] = nbr;
                        if (rank == 0)
                        {
                            nbr_size_new[warp_id] += group.size();
                        }
                    }
                    __syncwarp(__activemask());
                }
                __syncwarp();
                if (lane_id == 0)
                {
                    tries.values_[i][modified_gidx[warp_id][j] * 2 + 1] = nbr_size_new[warp_id];
                    if (nbr_size_new[warp_id] == 0)
                    {
                        tries.keys_[i][modified_gidx[warp_id][j]] = UINT32_MAX;
                    }
                }
                __syncwarp();
            }
        }
    }
}

__global__ void arrayAdd(
    const uint32_t *a,
    const uint32_t *b,
    uint32_t *c,
    const uint32_t size
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t num_threads = blockDim.x * gridDim.x;
    for (uint32_t i = tid; i < size; i += num_threads)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void writeSizesIntoArray(
    const uint32_t *in,
    uint32_t *out,
    const uint32_t size
) {
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
    {
        out[tid] = in [tid * 2 + 1];
    }
}