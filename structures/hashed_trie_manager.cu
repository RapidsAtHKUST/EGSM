#include <cstdint>
#include <vector>
#include <cub/cub.cuh>

#include "utils/cuda_helpers.h"
#include "utils/types.h"
#include "utils/globals.h"
#include "graph/graph.h"
#include "graph/graph_gpu.h"
#include "graph/operations.h"
#include "structures/hashed_tries.h"
#include "structures/hashed_trie_manager.h"
#include "structures/hashed_trie_manager_kernel.h"


HashedTrieManager::HashedTrieManager(
    const Graph& query, 
    const GraphGPU& query_gpu,
    const GraphGPU& data,
    HashedTries &tries)
: query_(query)
, q_edges_(NUM_EQ * 2)

, h_buffer_()
, h_num_candidates_(nullptr)
, h_compacted_vs_sizes_(nullptr)
, h_num_buckets_(nullptr)
, h_hash_table_offs_(nullptr)

, cardinalities_()

, compacted_vs_temp_(nullptr)
, C_()
, key_flags_{nullptr, nullptr}
, neighbor_flags_{nullptr, nullptr}
{
    // 1. build q_edges
    uint8_t edge_pos = 0u;
    for (uint32_t i = 0u; i < NUM_VQ * NUM_VQ; i++)
    {
        if (EIDX[i] != UINT8_MAX)
        {
            q_edges_[edge_pos++] = {i / NUM_VQ, i % NUM_VQ};
        }
    }

    // 2. allocate buffers.
    const uint32_t NUM_TRIES = NUM_EQ * 2 + 1;
    cudaErrorCheck(cudaMalloc(&tries.buffer_, sizeof(uint32_t) * NUM_TRIES * 4));
    tries.num_candidates_     = tries.buffer_;
    tries.compacted_vs_sizes_ = tries.num_candidates_       + NUM_TRIES;
    tries.num_buckets_        = tries.compacted_vs_sizes_   + NUM_TRIES;
    tries.hash_table_offs_    = tries.num_buckets_          + NUM_TRIES;
    cudaErrorCheck(cudaMemset(tries.num_candidates_, 0u, sizeof(uint32_t) * NUM_VQ));

    cudaErrorCheck(cudaMalloc(&compacted_vs_temp_, sizeof(uint32_t) * MAX_L_FREQ * NUM_EQ * 2));
    cudaErrorCheck(cudaMalloc(&tries.compacted_vs_, sizeof(uint32_t) * MAX_L_FREQ * NUM_EQ * 2));

    h_num_candidates_       = h_buffer_;
    h_compacted_vs_sizes_   = h_num_candidates_     + NUM_TRIES;
    h_num_buckets_          = h_compacted_vs_sizes_ + NUM_TRIES;
    h_hash_table_offs_      = h_num_buckets_        + NUM_TRIES;

    // 3. construct query nlf.
    uint32_t *query_nlf;
    uint32_t h_query_nlf[NUM_VQ * NUM_LQ];
    memset(h_query_nlf, 0u, sizeof(uint32_t) * NUM_VQ * NUM_LQ);

    cudaErrorCheck(cudaMalloc(&query_nlf, sizeof(uint32_t) * NUM_VQ * NUM_LQ));

    for (uint32_t u = 0u; u < NUM_VQ; u++)
    {
        for (uint32_t offset = query_.offsets_[u]; offset < query_.offsets_[u + 1]; offset ++)
        {
            const uint32_t u_other = query_.neighbors_[offset];

            h_query_nlf[u * NUM_LQ + query_.vlabels_[u_other]] ++;
        }
    }
    cudaErrorCheck(cudaMemcpy(query_nlf, h_query_nlf, sizeof(uint32_t) * NUM_VQ * NUM_LQ, cudaMemcpyHostToDevice));


    // 4. get the candidate vertices for each query vertex. 
    uint32_t *progress;
    cudaErrorCheck(cudaMalloc(&progress, sizeof(uint32_t) * 3));
    cudaErrorCheck(cudaMemset(progress, 0u, sizeof(uint32_t) * 3));

    // flag is a 2d bit matrix, flag[u][v] is set to 1 if the data vertex v
    // is a candidate of the query vertex u.
    const uint32_t NUM_FLAGS = DIV_CEIL(NUM_VD, 32) + 1;
    uint32_t *flags;
    cudaErrorCheck(cudaMalloc(&flags, sizeof(uint32_t) * NUM_FLAGS * NUM_VQ));
    cudaErrorCheck(cudaDeviceSynchronize());

    compareNLF<<<GRID_DIM, BLOCK_DIM>>>(
        query_gpu, data, query_nlf, 
        progress, flags, NUM_FLAGS, 
        compacted_vs_temp_, tries.num_candidates_);
    cudaErrorCheck(cudaDeviceSynchronize());

    // 5. record meta.
    cudaErrorCheck(cudaMemcpy(h_num_candidates_, tries.num_candidates_, sizeof(uint32_t) * NUM_VQ, cudaMemcpyDeviceToHost));

    for (uint32_t i = 0u; i < NUM_EQ * 2; i++)
    {
        auto [u, u_other] = q_edges_[i];
        h_compacted_vs_sizes_[i] = h_num_candidates_[u];
    }
    h_hash_table_offs_[0] = 0u;
    for (uint32_t i = 0; i < NUM_EQ * 2; i++)
    {
        h_num_buckets_[i] = DIV_CEIL(h_compacted_vs_sizes_[i] * CUCKOO_SCALE_PER_TABLE, BUCKET_DIM);
        // each bucket contains BUCKET_DIM cells, we need one more bucket for prefix sum.
        h_hash_table_offs_[i + 1] = h_hash_table_offs_[i] + h_num_buckets_[i] * BUCKET_DIM;
    }

    uint32_t total_size_per_table = h_hash_table_offs_[NUM_EQ * 2];


    // 6. allocate hash tables
    for (uint32_t i = 0; i < 2; i++)
    {
        cudaErrorCheck(cudaMalloc(&tries.keys_[i], sizeof(uint32_t) * total_size_per_table));
        cudaErrorCheck(cudaMalloc(&key_flags_[i], sizeof(uint32_t) * total_size_per_table));
        cudaErrorCheck(cudaMemset(key_flags_[i], 0u, sizeof(uint32_t) * total_size_per_table));
        // an additional value in the end for prefix-sum
        cudaErrorCheck(cudaMalloc(&tries.values_[i], sizeof(uint32_t) * (total_size_per_table * 2 + 1)));
        cudaErrorCheck(cudaMemset(tries.values_[i], 0u, sizeof(uint32_t) * (total_size_per_table * 2 + 1)));
    }

    // 7. build hash keys
    std::unordered_map<uint32_t, uint32_t> finished_built_u;
    const uint32_t PRIME = 4294967291u;
    std::mt19937 gen(0);
    std::uniform_int_distribution<uint32_t> distrib(0u, UINT32_MAX);
    uint32_t success = 0u;

    for (uint32_t i = 0u; i < NUM_EQ * 2; i++)
    {
        auto [u, u_other] = q_edges_[i];

        if (finished_built_u.find(u) == finished_built_u.end())
        {
            do {
                cudaErrorCheck(cudaMemset(progress, 0u, sizeof(uint32_t) * 3));
                for(uint32_t j = 0; j < 2; j++)
                {
                    cudaErrorCheck(cudaMemset(
                        tries.keys_[j] + h_hash_table_offs_[i], 
                        UINT32_MAX, 
                        sizeof(uint32_t) * h_num_buckets_[i] * BUCKET_DIM
                    ));
                }
                GenerateCs(tries, i, distrib, gen, PRIME);
                cudaErrorCheck(cudaDeviceSynchronize());

                buildHashKeys<<<GRID_DIM, BLOCK_DIM>>>(
                    compacted_vs_temp_ + MAX_L_FREQ * u,
                    h_compacted_vs_sizes_[i],
                    tries.keys_[0] + h_hash_table_offs_[i],
                    tries.keys_[1] + h_hash_table_offs_[i],
                    h_num_buckets_[i],
                    C_[tries.CIdx(i, 0, 0)],
                    C_[tries.CIdx(i, 0, 1)],
                    C_[tries.CIdx(i, 1, 0)],
                    C_[tries.CIdx(i, 1, 1)],
                    progress, 
                    progress + 2
                );
                cudaErrorCheck(cudaMemcpy(&success, progress + 2, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            }
            while (success != 0u);
            finished_built_u[u] = i;
        }
        else
        {
            uint32_t reference_eidx = finished_built_u[u];
            for (uint32_t j = 0u; j < 2; j++)
            {
                C_[tries.CIdx(i, j, 0)] = C_[tries.CIdx(reference_eidx, j, 0)];
                C_[tries.CIdx(i, j, 1)] = C_[tries.CIdx(reference_eidx, j, 1)];
                cudaErrorCheck(cudaMemcpy(
                    tries.keys_[j] + h_hash_table_offs_[i],
                    tries.keys_[j] + h_hash_table_offs_[reference_eidx],
                    sizeof(uint32_t) * h_num_buckets_[reference_eidx] * BUCKET_DIM,
                    cudaMemcpyDeviceToDevice
                ));
            }
        }
    }

    // 8. count the number of neighbors for each vertex
    for (uint32_t i = 0u; i < NUM_EQ * 2; i++)
    {
        auto [u, u_other] = q_edges_[i];

        for (uint32_t j = 0; j < 2; j++)
        {
            cudaErrorCheck(cudaMemset(progress, 0u, sizeof(uint32_t) * 3));
            cudaErrorCheck(cudaDeviceSynchronize());

            buildHashValuesCount<<<GRID_DIM, BLOCK_DIM>>>(
                data,
                flags + NUM_FLAGS * u_other,
                NUM_FLAGS,
                progress,
                tries.keys_[j] + h_hash_table_offs_[i],
                tries.values_[j] + h_hash_table_offs_[i] * 2,
                h_num_buckets_[i]
            );
            cudaErrorCheck(cudaDeviceSynchronize());
        }
    }
    for (uint32_t i = 0; i < 2; i++)
    {
        void *d_temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes, 
            tries.values_[i],
            tries.values_[i],
            total_size_per_table * 2 + 1);

        cudaErrorCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_bytes, 
            tries.values_[i],
            tries.values_[i],
            total_size_per_table * 2 + 1);
        cudaErrorCheck(cudaDeviceSynchronize());
        uint32_t total_nbrs = 0u;
        cudaErrorCheck(cudaMemcpy(
            &total_nbrs, 
            tries.values_[i] + total_size_per_table * 2,
            sizeof(uint32_t),
            cudaMemcpyDeviceToHost
        ));
        cudaErrorCheck(cudaMalloc(&tries.neighbors_[i], sizeof(uint32_t) * total_nbrs));
        cudaErrorCheck(cudaMalloc(&neighbor_flags_[i], sizeof(uint32_t) * total_nbrs));
        cudaErrorCheck(cudaMemset(neighbor_flags_[i], 0u, sizeof(uint32_t) * total_nbrs));

        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaFree(d_temp_storage));
    }
    for (uint32_t i = 0u; i < NUM_EQ * 2; i++)
    {
        auto [u, u_other] = q_edges_[i];

        for (uint32_t j = 0; j < 2; j++)
        {
            cudaErrorCheck(cudaMemset(progress, 0u, sizeof(uint32_t) * 3));
            cudaErrorCheck(cudaDeviceSynchronize());

            buildHashValuesWrite<<<GRID_DIM, BLOCK_DIM>>>(
                data,
                flags + NUM_FLAGS * u_other,
                NUM_FLAGS,
                progress,
                tries.keys_[j] + h_hash_table_offs_[i],
                tries.values_[j] + h_hash_table_offs_[i] * 2,
                h_num_buckets_[i],
                tries.neighbors_[j]
            );
            cudaErrorCheck(cudaDeviceSynchronize());
        }
    }

    cudaErrorCheck(cudaMemcpy(tries.buffer_, h_buffer_, sizeof(uint32_t) * NUM_TRIES * 4, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C, C_, sizeof(uint32_t) * MAX_ECOUNT * 2 * 2 * 2));
    //Print();

    cudaErrorCheck(cudaFree(flags));
    cudaErrorCheck(cudaFree(progress));
    cudaErrorCheck(cudaFree(query_nlf));
}

void HashedTrieManager::GenerateCs(
    const HashedTries &tries,
    const uint32_t index,
    std::uniform_int_distribution<uint32_t>& distrib,
    std::mt19937& gen,
    const uint32_t PRIME
) {
    for (uint32_t i = 0; i < 2; i++)
    {
        C_[tries.CIdx(index, i, 0)] = distrib(gen) % PRIME;
        C_[tries.CIdx(index, i, 0)] = std::max(1u, C_[tries.CIdx(index, i, 0)]);
        C_[tries.CIdx(index, i, 1)] = distrib(gen) % PRIME;
    }
}

void HashedTrieManager::Filter(
    HashedTries &tries, 
    bool filtering_3rd, 
    uint32_t filtering_order_start_v
) {
    if (filtering_3rd)
    {
        uint32_t degeneracy_order[NUM_VQ];
        uint32_t degeneracy_offset[NUM_VQ + 1];
        uint32_t degeneracy_neighbors[NUM_EQ * 2 + 1];

        // compute degeneracy order and its offsets and neighbors
        if (filtering_order_start_v == UINT32_MAX)
        {
            computeDegeneracyOrder(query_, degeneracy_order);
        }
        else
        {
            computeBFSOrder(query_, degeneracy_order, filtering_order_start_v);
        }
        
        degeneracy_offset[0] = 0u;
        for (uint32_t i = 0; i < NUM_VQ; i++)
        {
            degeneracy_offset[i + 1] = degeneracy_offset[i];
            uint32_t u = degeneracy_order[i];
            for (uint32_t j = 0; j < NUM_VQ; j++)
            {
                uint32_t u_other = degeneracy_order[j];
                if (std::binary_search(
                    &query_.neighbors_[query_.offsets_[u]], 
                    &query_.neighbors_[query_.offsets_[u + 1]], 
                    u_other
                )) {
                    degeneracy_neighbors[degeneracy_offset[i + 1]++] = u_other;
                }
            }
        }

        // outer-inner pruning
        uint32_t *progress;
        cudaErrorCheck(cudaMalloc(&progress, sizeof(uint32_t)));

        for (uint32_t i = 0; i < NUM_VQ; i++)
        {
            const uint32_t u = degeneracy_order[i];
            if (degeneracy_offset[i + 1] - degeneracy_offset[i] <= 1) continue;

            for (uint32_t offset = degeneracy_offset[i] + 1; offset < degeneracy_offset[i + 1]; offset++)
            {
                const uint32_t u_other_pre = degeneracy_neighbors[offset - 1];
                const uint32_t u_other = degeneracy_neighbors[offset];

                cudaErrorCheck(cudaMemset(progress, 0u, sizeof(uint32_t)));
                cudaErrorCheck(cudaDeviceSynchronize());

                semiJoin<<<GRID_DIM, BLOCK_DIM>>>(
                    key_flags_[0],
                    key_flags_[1],
                    neighbor_flags_[0],
                    neighbor_flags_[1],
                    EIDX[u * (NUM_VQ) + u_other_pre],
                    EIDX[u * (NUM_VQ) + u_other],
                    EIDX[u_other * (NUM_VQ) + u],
                    progress
                );
                cudaErrorCheck(cudaDeviceSynchronize());
            }
        }
        // inner-outer pruning
        for (long i = NUM_VQ - 1; i >= 0; i--)
        {
            uint32_t u = degeneracy_order[i];
            if (degeneracy_offset[i + 1] - degeneracy_offset[i] <= 1) continue;

            for (long offset = degeneracy_offset[i + 1] - 2; offset >= degeneracy_offset[i]; offset--)
            {
                const uint32_t u_other_pre = degeneracy_neighbors[offset + 1];
                const uint32_t u_other = degeneracy_neighbors[offset];

                cudaErrorCheck(cudaMemset(progress, 0u, sizeof(uint32_t)));
                cudaErrorCheck(cudaDeviceSynchronize());

                semiJoin<<<GRID_DIM, BLOCK_DIM>>>(
                    key_flags_[0],
                    key_flags_[1],
                    neighbor_flags_[0],
                    neighbor_flags_[1],
                    EIDX[u * (NUM_VQ) + u_other_pre],
                    EIDX[u * (NUM_VQ) + u_other],
                    EIDX[u_other * (NUM_VQ) + u],
                    progress
                );
                cudaErrorCheck(cudaDeviceSynchronize());
            }
        }

        // compact neighbors
        cudaErrorCheck(cudaMemset(progress, 0u, sizeof(uint32_t)));
        cudaErrorCheck(cudaDeviceSynchronize());

        compactMiddleLevel<<<GRID_DIM, BLOCK_DIM>>>(*this, tries, progress);
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaFree(progress));
    }

    // compact the first level of each trie
    NotUintMax not_uint_max{};
    uint32_t *num_selected_out_temp, h_num_selected_out_temp[MAX_ECOUNT * 2];
    cudaErrorCheck(cudaMalloc(&num_selected_out_temp, sizeof(uint32_t) * NUM_EQ * 2));

    for (uint32_t i = 0u; i < NUM_EQ * 2; i++)
    {
        auto [u, u_other] = q_edges_[i];
        // compact all candidates in keys to compacted_vs_
        // and update h_compacted_vs_sizes_ and compacted_vs_sizes_
        cudaErrorCheck(cudaDeviceSynchronize());
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceSelect::If(
            d_temp_storage, temp_storage_bytes, 
            tries.keys_[0] + h_hash_table_offs_[i], 
            compacted_vs_temp_ + i * MAX_L_FREQ,
            num_selected_out_temp + i, 
            h_num_buckets_[i] * BUCKET_DIM, not_uint_max);
        // Allocate temporary storage
        cudaErrorCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        // Run selection
        cub::DeviceSelect::If(
            d_temp_storage, temp_storage_bytes, 
            tries.keys_[0] + h_hash_table_offs_[i], 
            compacted_vs_temp_ + i * MAX_L_FREQ,
            num_selected_out_temp + i, 
            h_num_buckets_[i] * BUCKET_DIM, not_uint_max);
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaFree(d_temp_storage));
    }
    cudaErrorCheck(cudaMemcpy(
        h_num_selected_out_temp, num_selected_out_temp,
        sizeof(uint32_t) * NUM_EQ * 2,
        cudaMemcpyDeviceToHost));

    for (uint32_t i = 0u; i < NUM_EQ * 2; i++)
    {
        auto [u, u_other] = q_edges_[i];
        // compact all candidates in keys to compacted_vs_
        // and update h_compacted_vs_sizes_ and compacted_vs_sizes_
        cudaErrorCheck(cudaDeviceSynchronize());
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceSelect::If(
            d_temp_storage, temp_storage_bytes, 
            tries.keys_[1] + h_hash_table_offs_[i], 
            compacted_vs_temp_ + i * MAX_L_FREQ + h_num_selected_out_temp[i],
            tries.compacted_vs_sizes_ + i, 
            h_num_buckets_[i] * BUCKET_DIM, not_uint_max);
        // Allocate temporary storage
        cudaErrorCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        // Run selection
        cub::DeviceSelect::If(
            d_temp_storage, temp_storage_bytes, 
            tries.keys_[1] + h_hash_table_offs_[i], 
            compacted_vs_temp_ + i * MAX_L_FREQ + h_num_selected_out_temp[i],
            tries.compacted_vs_sizes_ + i, 
            h_num_buckets_[i] * BUCKET_DIM, not_uint_max);
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaFree(d_temp_storage));
    }
    // copy compacted_vs_sizes to host
    arrayAdd<<<GRID_DIM, BLOCK_DIM>>>(
        tries.compacted_vs_sizes_,
        num_selected_out_temp,
        tries.compacted_vs_sizes_,
        NUM_EQ * 2
    );
    cudaErrorCheck(cudaMemcpy(
        h_compacted_vs_sizes_, tries.compacted_vs_sizes_,
        sizeof(uint32_t) * NUM_EQ * 2,
        cudaMemcpyDeviceToHost));
    // sort all compacted_vs
    for (uint32_t i = 0; i < NUM_EQ * 2; i++)
    {
        void     *d_temp_storage = NULL;
        size_t   temp_storage_bytes = 0;
        cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes, 
            compacted_vs_temp_ + i * MAX_L_FREQ, 
            tries.compacted_vs_ + i * MAX_L_FREQ, 
            h_compacted_vs_sizes_[i]);
        // Allocate temporary storage
        cudaErrorCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
        // Run sorting operation
        cub::DeviceRadixSort::SortKeys(
            d_temp_storage, temp_storage_bytes,  
            compacted_vs_temp_ + i * MAX_L_FREQ, 
            tries.compacted_vs_ + i * MAX_L_FREQ, 
            h_compacted_vs_sizes_[i]);
        cudaErrorCheck(cudaDeviceSynchronize());
        cudaErrorCheck(cudaFree(d_temp_storage));
    }
    cudaErrorCheck(cudaDeviceSynchronize());
    cudaErrorCheck(cudaFree(num_selected_out_temp));
}

uint8_t HashedTrieManager::GetFirstMinOff(const uint32_t first_u) const
{
    uint8_t first_u_min_off;
    uint32_t min_compacted_size = UINT32_MAX;
    for (uint32_t offset = query_.offsets_[first_u]; offset < query_.offsets_[first_u + 1]; offset++)
    {
        const uint32_t u_other = query_.neighbors_[offset];
        const uint32_t local_off = EIDX[first_u * NUM_VQ + u_other];
        if (h_compacted_vs_sizes_[local_off] < min_compacted_size)
        {
            min_compacted_size = h_compacted_vs_sizes_[local_off];
            first_u_min_off = local_off;
        }
    }
    return first_u_min_off;
}

void HashedTrieManager::GetCardinalities(const HashedTries &tries)
{
    uint32_t h_total, *total;
    cudaErrorCheck(cudaMalloc(&total, sizeof(uint32_t)));
    for (uint32_t i = 0; i < NUM_EQ * 2; i++)
    {
        cardinalities_[i] = 0u;
        for (uint32_t j = 0; j < 2; j++)
        {
            uint32_t *num_nbrs;
            cudaErrorCheck(cudaMalloc(&num_nbrs, sizeof(uint32_t) * h_num_buckets_[i] * BUCKET_DIM));
            writeSizesIntoArray<<<h_num_buckets_[i] * BUCKET_DIM / WARP_PER_BLOCK, BLOCK_DIM>>>(
                tries.values_[j] + h_hash_table_offs_[i] * 2, num_nbrs, h_num_buckets_[i] * BUCKET_DIM
            );
            cudaErrorCheck(cudaDeviceSynchronize());
            void     *d_temp_storage = NULL;
            size_t   temp_storage_bytes = 0;
            cub::DeviceReduce::Sum(
                d_temp_storage, temp_storage_bytes,
                num_nbrs, total, h_num_buckets_[i] * BUCKET_DIM
            );
            cudaErrorCheck(cudaMalloc(&d_temp_storage, temp_storage_bytes));
            cub::DeviceReduce::Sum(
                d_temp_storage, temp_storage_bytes,
                num_nbrs, total, h_num_buckets_[i] * BUCKET_DIM
            );
            cudaErrorCheck(cudaDeviceSynchronize());

            cudaErrorCheck(cudaMemcpy(&h_total, total, sizeof(uint32_t), cudaMemcpyDeviceToHost));
            cardinalities_[i] += h_total;

            cudaErrorCheck(cudaFree(d_temp_storage));
            cudaErrorCheck(cudaFree(num_nbrs));
        }
    }
    cudaErrorCheck(cudaFree(total));
}

void HashedTrieManager::Print()
{
    std::cout << "\n# Candidate edges: \n";
    for (uint32_t i = 0; i < NUM_EQ * 2; i++)
    {
        if (q_edges_[i].first > q_edges_[i].second) continue;
        std::cout << '(' << q_edges_[i].first << ", " << q_edges_[i].second << "): "
            << cardinalities_[i] << ' ' 
            << q_edges_[i].first << ": " << h_compacted_vs_sizes_[i] << ' '
            << q_edges_[i].second << ": " << h_compacted_vs_sizes_[EIDX[q_edges_[i].second * NUM_VQ + q_edges_[i].first]] << '\n';
    }
    std::cout << '\n';
}

void HashedTrieManager::Deallocate()
{
    for (uint32_t i = 0; i < 2; i++)
    {
        cudaErrorCheck(cudaFree(neighbor_flags_[i]));
        cudaErrorCheck(cudaFree(key_flags_[i]));
    }
    cudaErrorCheck(cudaFree(compacted_vs_temp_));
}

void HashedTrieManager::DeallocateTries(HashedTries &tries)
{
    for (uint32_t i = 0; i < 2; i++)
    {
        cudaErrorCheck(cudaFree(tries.neighbors_[i]));
        cudaErrorCheck(cudaFree(tries.values_[i]));
        cudaErrorCheck(cudaFree(tries.keys_[i]));
    }
    cudaErrorCheck(cudaFree(tries.compacted_vs_));
    cudaErrorCheck(cudaFree(tries.buffer_));
}
