#include <cstdint>

#include "utils/types.h"
#include "utils/globals.h"
#include "structures/hashed_tries.h"


__constant__ HashedTries tries;

HashedTries::HashedTries() {}

__device__ uint32_t HashedTries::BIdx(
    uint32_t table_index, 
    uint32_t element_index
) const {
    return hash_table_offs_[table_index] + element_index;
}

__host__ __device__ uint32_t HashedTries::CIdx(
    uint32_t trie_index, 
    uint32_t table_index, 
    uint32_t const_index
) const {
    return trie_index * 4 + table_index * 2 + const_index;
}

__device__ uint2 HashedTries::HashSearch(
    uint32_t table_index, 
    const uint32_t key
) const {
    Bucket bucket;
    uint2 result;
    uint32_t hash_value, *bucket_uint32;
    for (uint32_t i = 0u; i < 2u; i++)
    {
        hash_value = (C[CIdx(table_index, i, 0)] ^ key +
            C[CIdx(table_index, i, 1)]) % num_buckets_[table_index];

        bucket = reinterpret_cast<Bucket*>(keys_[i])
            [hash_table_offs_[table_index] / BUCKET_DIM + hash_value];
        bucket_uint32 = reinterpret_cast<uint32_t*>(&bucket);

        for (uint32_t j = 0; j < BUCKET_DIM; j++)
        {
            if (bucket_uint32[j] == key)
            {
                result.y = i;
                result.x = hash_value * BUCKET_DIM + j;
                return result;
            }
        }
    }
    result.y = 0u;
    result.x = UINT32_MAX;
    return result;
}
