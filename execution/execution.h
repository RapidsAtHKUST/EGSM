#ifndef EXECUTION_EXECUTION_H
#define EXECUTION_EXECUTION_H


#include <cstdint>

#include "utils/mem_pool.h"
#include "graph/graph.h"
#include "processing/plan.h"
#include "structures/hashed_tries.h"
#include "structures/hashed_trie_manager.h"


void copyConfig(
    const bool adaptive_ordering,
    const bool load_balancing
);

void copyGraphMeta(
    const Graph& query_graph,
    const Graph& data_graph,
    const GraphUtils& query_utils
);
void copyTries(const HashedTries& hashed_tries);

void matchDFSGroup(
    const HashedTrieManager& manager,
    Plan& plan,
    MemPool& pool,
    PoolElem& res,
    unsigned long long int& res_size
);

#endif //EXECUTION_EXECUTION_H
