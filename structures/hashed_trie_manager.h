#ifndef STRUCTURES_HASHED_TRIE_MANAGER_H
#define STRUCTURES_HASHED_TRIE_MANAGER_H

#include <cstdint>
#include <random>
#include <vector>

#include "utils/config.h"

#include "graph/graph.h"
#include "graph/graph_gpu.h"
#include "structures/hashed_tries.h"

struct HashedTrieManager
{
    const Graph &query_;
    std::vector<std::pair<uint32_t, uint32_t>> q_edges_;

    // a copy of the meta data of the hashed trie on host
    uint32_t h_buffer_[(MAX_ECOUNT * 2 + 1) * 4];
    uint32_t *h_num_candidates_;
    uint32_t *h_compacted_vs_sizes_;
    uint32_t *h_num_buckets_;
    uint32_t *h_hash_table_offs_;

    uint32_t cardinalities_[MAX_ECOUNT * 2];

    uint32_t *compacted_vs_temp_;
    uint32_t C_[MAX_ECOUNT * 2 * 2 * 2];
    uint32_t *key_flags_[2];
    uint32_t *neighbor_flags_[2];

    HashedTrieManager(
        const Graph &query,
        const GraphGPU &query_gpu,
        const GraphGPU &data,
        HashedTries &tries);

    void GenerateCs(
        const HashedTries &tries,
        const uint32_t index,
        std::uniform_int_distribution<uint32_t> &distrib,
        std::mt19937 &gen,
        const uint32_t PRIME);

    void Filter(
        HashedTries &tries, 
        const bool filtering_3rd, 
        const uint32_t filtering_order_start_v);

    uint8_t GetFirstMinOff(const uint32_t first_u) const;
    void GetCardinalities(const HashedTries &tries);
    void Print();

    void Deallocate();
    void DeallocateTries(HashedTries &tries);
};

#endif // STRUCTURES_HASHED_TRIES_H
