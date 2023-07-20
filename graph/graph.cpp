#include <algorithm>
#include <cstdint>
#include <fstream>
#include <unordered_map>

#include "graph/graph.h"


Graph::Graph(std::string path, std::unordered_map<uint32_t, uint32_t>& label_map)
: vcount_(0)
, ecount_(0)

, vlabels_(nullptr)
, vdegs_(nullptr)
, lcount_(0u)
, lfreq_max_(0u)
, deg_max_(0u)

, offsets_(nullptr)
, neighbors_(nullptr)

, vlabel_freq_()
, elabel_freq_()
{
    std::ifstream ifs(path);
    if(ifs.fail())
    {
        std::cout << "File not exist!\n";
        exit(-1);
    }

    // true for the query graph, false for the data graph
    bool set_label_map = label_map.empty();

    char type;
    ifs >> type >> vcount_ >> ecount_;
    if (set_label_map && (vcount_ > 16 || ecount_ > 32))
    {
        std::cout << "The query graph should have at most 16 vertices and 32 edges.\n";
        exit(-1);
    }

    vlabels_ = new uint32_t[vcount_];
    vdegs_ = new uint32_t[vcount_];
    offsets_ = new uint32_t[vcount_ + 1];
    offsets_[0] = 0u;
    neighbors_ = new uint32_t[ecount_ * 2 + 1];

    uint32_t* neighbors_offset = new uint32_t[vcount_]();

    while (ifs >> type)
    {
        if (type == 'v')
        {
            uint32_t vertex_id, degree;
            uint32_t label;
            ifs >> vertex_id >> label >> degree;
            if (set_label_map)
            {
                if (label_map.find(label) == label_map.end())
                {
                    uint32_t label_map_size = label_map.size();
                    label_map[label] = label_map_size;
                }
                label = label_map.at(label);
            }
            else
            {
                if (label_map.find(label) == label_map.end())
                {
                    label = label_map.size();
                }
                else
                {
                    label = label_map.at(label);
                }
            }

            vlabels_[vertex_id] = label;
            if (label < label_map.size())
            {
                if (vlabel_freq_.find(label) == vlabel_freq_.end())
                {
                    vlabel_freq_[label] = 0;
                }
                vlabel_freq_[label] += 1;
            }
            offsets_[vertex_id + 1] = offsets_[vertex_id] + degree;

            vdegs_[vertex_id] = degree;
            if (degree > deg_max_)
            {
                deg_max_ = degree;
            }
        }
        else
        {
            uint32_t from_id, to_id;
            ifs >> from_id >> to_id;
            
            uint32_t offset = offsets_[from_id] + neighbors_offset[from_id];
            neighbors_[offset] = to_id;

            offset = offsets_[to_id] + neighbors_offset[to_id];
            neighbors_[offset] = from_id;

            neighbors_offset[from_id]++;
            neighbors_offset[to_id]++;

            if (elabel_freq_.find({vlabels_[from_id], vlabels_[to_id]}) == elabel_freq_.end())
            {
                elabel_freq_[{vlabels_[from_id], vlabels_[to_id]}] = 0;
            }
            elabel_freq_[{vlabels_[from_id], vlabels_[to_id]}] ++;
            if (elabel_freq_.find({vlabels_[to_id], vlabels_[from_id]}) == elabel_freq_.end())
            {
                elabel_freq_[{vlabels_[to_id], vlabels_[from_id]}] = 0;
            }
            elabel_freq_[{vlabels_[to_id], vlabels_[from_id]}] ++;
        }
    }
    ifs.close();

    lcount_ = std::max_element(vlabel_freq_.begin(), vlabel_freq_.end(), 
        [](const auto& v1, const auto& v2)
        {
            return v1.first < v2.first;
        }
    )->first + 1;
    lfreq_max_ = std::max_element(vlabel_freq_.begin(), vlabel_freq_.end(), 
        [](const auto& v1, const auto& v2)
        {
            return v1.second < v2.second;
        }
    )->second;

    delete[] neighbors_offset;
}

Graph::~Graph()
{
    delete[] vlabels_;
    delete[] vdegs_;
    delete[] offsets_;
    delete[] neighbors_;
}


void GraphUtils::Set(const Graph& g)
{
    std::fill(eidx_, eidx_ + g.vcount_ * g.vcount_, UINT8_MAX);
    uint8_t edge_pos = 0u;

    for (uint32_t u = 0u; u < g.vcount_; u++)
    {
        for (uint32_t offset = g.offsets_[u]; offset < g.offsets_[u + 1]; offset ++)
        {
            const uint32_t u_other = g.neighbors_[offset];

            uint32_t key = u * (g.vcount_) + u_other;
            eidx_[key] = edge_pos;
            edge_pos++;
        }
    }

    for (uint32_t i = 0; i < g.vcount_; i++)
    {
        nbrbits_[i] = 0u;
        uint8_t *based = eidx_ + i * g.vcount_;
        for (uint32_t j = 0; j < g.vcount_; j++)
        {
            if (based[j] != UINT8_MAX)
            {
                nbrbits_[i] |= (1 << j);
            }
        }
    }
}
