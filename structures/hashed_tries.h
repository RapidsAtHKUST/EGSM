#ifndef STRUCTURES_HASHED_TRIES_H
#define STRUCTURES_HASHED_TRIES_H

#include <cstdint>


struct HashedTries
{
    // buffer_ is used for num_candidates_,
    // compacted_vs_sizes_, compacted_vs_offs_,
    // and hash_keys_offs_.
    uint32_t *buffer_;
    uint32_t *num_candidates_;
    uint32_t *compacted_vs_sizes_;
    uint32_t *num_buckets_;
    uint32_t *hash_table_offs_;

    // hash tables
    uint32_t *compacted_vs_;
    uint32_t *keys_[2];
    uint32_t *values_[2];
    uint32_t *neighbors_[2];

    HashedTries();

    __device__ uint32_t BIdx(
        uint32_t table_index, 
        uint32_t element_index) const;

    __host__ __device__ uint32_t CIdx(
        uint32_t trie_index, 
        uint32_t table_index, 
        uint32_t const_index) const;

    __device__ uint2 HashSearch(
        uint32_t table_index, 
        const uint32_t key) const;
};

extern __constant__ HashedTries tries;

#endif //STRUCTURES_HASHED_TRIES_H
