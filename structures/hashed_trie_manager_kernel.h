#ifndef STRUCTURES_HASHED_TRIE_MANAGER_KERNEL_H
#define STRUCTURES_HASHED_TRIE_MANAGER_KERNEL_H

#include <cstdint>

#include "graph/graph.h"
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
);

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
);

__global__ void buildHashValuesCount(
    const GraphGPU data,
    const uint32_t *flags_second_level,
    const uint32_t num_flags,
    uint32_t *progress,
    uint32_t *hash_keys,
    uint32_t *hash_values,
    const uint32_t num_bucket
);

__global__ void buildHashValuesWrite(
    const GraphGPU data,
    const uint32_t *flags_second_level,
    const uint32_t num_flags,
    uint32_t *progress,
    uint32_t *hash_keys,
    uint32_t *hash_values,
    const uint32_t num_bucket,
    uint32_t *neighbors
);

__global__ void semiJoin(
    uint32_t* manager_key_flags0,
    uint32_t* manager_key_flags1,
    uint32_t* manager_neighbor_flags0,
    uint32_t* manager_neighbor_flags1,
    const uint32_t lidx,
    const uint32_t ridx,
    const uint32_t reversed_ridx,
    uint32_t *progress
);

__global__ void compactMiddleLevel(
    const HashedTrieManager manager,
    HashedTries tries,
    uint32_t *progress
);

__global__ void arrayAdd(
    const uint32_t *a,
    const uint32_t *b,
    uint32_t *c,
    const uint32_t size
);

__global__ void writeSizesIntoArray(
    const uint32_t *in,
    uint32_t *out,
    const uint32_t size
);

#endif //STRUCTURES_HASHED_TRIE_MANAGER_KERNEL_H
