#ifndef PROCESSING_JOIN_BFS_H
#define PROCESSING_JOIN_BFS_H

#include <cstdint>

#include "structures/hashed_trie_manager.h"
#include "processing/plan.h"


void bfsOneLevel(
    uint32_t num_warps,
    uint32_t*& partial_results,
    uint32_t& partial_result_count,
    InitialOrder initial_order,
    uint8_t next_u,
    uint8_t depth,
    uint8_t min_off = UINT8_MAX /* need when depth == 1 */
);


#endif //PROCESSING_JOIN_BFS_H
