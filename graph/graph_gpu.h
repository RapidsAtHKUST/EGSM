#ifndef GRAPH_GRAPH_GPU_H
#define GRAPH_GRAPH_GPU_H

#include <cstdint>
#include <string>
#include <unordered_map>

// for pair hash function
#include "utils/config.h"
#include "utils/nucleus/nd.h"


class GraphGPU
{
public:
    uint32_t *vlabels_;
    uint32_t *offsets_;
    uint32_t *neighbors_;
public:
    GraphGPU(const Graph& g);
    void Deallocate();
};


#endif //GRAPH_GRAPH_GPU_H
