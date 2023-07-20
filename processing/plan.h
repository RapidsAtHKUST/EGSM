#ifndef PROCESSING_PLAN_H
#define PROCESSING_PLAN_H

#include <string>

#include "utils/config.h"
#include "graph/graph.h"
#include "utils/nucleus/nd_interface.h"

struct InitialOrder
{
    uint8_t u[MAX_VCOUNT];
};

struct Plan
{
    uint8_t start_vertex_id_;
    uint8_t mask_size_;
    uint16_t masks_[MAX_VCOUNT];
    uint8_t mask_size_prefix_sum_[MAX_VCOUNT];
    uint8_t res_pos[MAX_VCOUNT];
    uint16_t is_tree_group_;

    Plan() {}

    Plan(
        const Graph& query,
        uint32_t *relation_sizes,
        std::string method
    );

    Plan(
        const Graph& query,
        uint32_t *relation_sizes,
        uint32_t gsi
    );

    void AddGroup(uint8_t i, InitialOrder& initial_order);

    void Print(const Graph& query);
};

extern __constant__ Plan c_plan;


#endif //PROCESSING_PLAN_H
