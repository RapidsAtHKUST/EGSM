#include <algorithm>
#include <cstdint>
#include <fstream>
#include <unordered_map>

#include "utils/cuda_helpers.h"
#include "graph/graph.h"
#include "graph/graph_gpu.h"


GraphGPU::GraphGPU(const Graph& g)
: vlabels_(nullptr)
, offsets_(nullptr)
, neighbors_(nullptr)
{
    cudaErrorCheck(cudaMalloc(&vlabels_, sizeof(uint32_t) * g.vcount_));
    cudaErrorCheck(cudaMalloc(&offsets_, sizeof(uint32_t) * (g.vcount_ + 1)));
    cudaErrorCheck(cudaMalloc(&neighbors_, sizeof(uint32_t) * (g.ecount_ * 2 + 1)));
    cudaErrorCheck(cudaMemcpy(vlabels_, g.vlabels_, sizeof(uint32_t) * g.vcount_, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(offsets_, g.offsets_, sizeof(uint32_t) * (g.vcount_ + 1), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpy(neighbors_, g.neighbors_, sizeof(uint32_t) * (g.ecount_ * 2 + 1), cudaMemcpyHostToDevice));
}


void GraphGPU::Deallocate()
{
    cudaErrorCheck(cudaFree(vlabels_));
    cudaErrorCheck(cudaFree(offsets_));
    cudaErrorCheck(cudaFree(neighbors_));
}
