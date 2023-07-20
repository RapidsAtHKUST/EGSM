#ifndef GRAPH_OPERATIONS_H
#define GRAPH_OPERATIONS_H

#include <cstdint>
#include "graph/graph.h"


void computeDegeneracyOrder(
    const Graph& query, uint32_t *degeneracy_order
);

void computeBFSOrder(
    const Graph& query, uint32_t *bfs_order, const uint32_t start_v
);

#endif //GRAPH_OPERATIONS_H
