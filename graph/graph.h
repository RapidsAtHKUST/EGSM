#ifndef GRAPH_GRAPH_H
#define GRAPH_GRAPH_H

#include <cstdint>
#include <string>
#include <unordered_map>

// for pair hash function
#include "utils/config.h"
#include "utils/nucleus/nd.h"


class Graph
{
public:
    uint32_t vcount_;
    uint32_t ecount_;

    uint32_t *vlabels_;
    uint32_t *vdegs_;
    uint32_t lcount_;
    uint32_t lfreq_max_;
    uint32_t deg_max_;

    uint32_t *offsets_;
    uint32_t *neighbors_;

    std::unordered_map<uint32_t, uint32_t> vlabel_freq_;
    std::unordered_map<std::pair<uint32_t, uint32_t>, uint32_t> elabel_freq_;
public:
    Graph(std::string path, std::unordered_map<uint32_t, uint32_t>& label_map);
    ~Graph();
};



class GraphUtils
{
public:
    uint8_t eidx_[MAX_VCOUNT * MAX_VCOUNT];
    uint16_t nbrbits_[MAX_VCOUNT];
public:
    void Set(const Graph& g);
};


#endif //GRAPH_GRAPH_H
