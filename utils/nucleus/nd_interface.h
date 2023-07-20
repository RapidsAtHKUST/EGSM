#ifndef __ND_INTERFACE_H
#define __ND_INTERFACE_H

#include <cstdint>
#include <vector>
#include "graph/graph.h"
#include "nd.h"

struct nd_tree_node {
    int id_;
    int parent_;
    std::vector<int> children_;
    std::vector<int> vertices_;
    uint32_t k_;
    uint32_t r_;
    uint32_t s_;
    uint32_t num_edges_;
    double density_;
};

class nd_interface {
public:
    static void convert_graph(const Graph& query_graph, nd_Graph &nd_graph);

    static void nd(const Graph& query_graph, uint32_t r, uint32_t s, vector<nd_tree_node> &nd_tree);
    static void nd(nd_Graph& graph, uint32_t r, uint32_t s, vector<nd_tree_node> &nd_tree);
    static void print_nd_tree(uint32_t r, uint32_t s, std::vector<nd_tree_node>& nd_tree);
};


#endif //__ND_INTERFACE_H
