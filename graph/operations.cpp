#include <cstdint>

#include "graph/graph.h"
#include "graph/operations.h"


void computeDegeneracyOrder(
    const Graph& query, uint32_t *degeneracy_order
) {
    uint32_t core_table[query.vcount_];       // core values.
    uint32_t vertices[query.vcount_];         // Vertices sorted by degree.
    uint32_t position[query.vcount_];         // The position of vertices in vertices array.
    uint32_t degree_bin[query.deg_max_ + 1];  // Degree from 0 to max_degree.
    uint32_t offset[query.deg_max_ + 1];      // The offset in vertices array according to degree.

    std::fill(degree_bin, degree_bin + (query.deg_max_ + 1), 0);

    for (int i = 0; i < query.vcount_; ++i)
    {
        int degree = query.vdegs_[i];
        core_table[i] = degree;
        degree_bin[degree] += 1;
    }

    int start = 0;
    for (int i = 0; i < query.deg_max_ + 1; ++i)
    {
        offset[i] = start;
        start += degree_bin[i];
    }

    for (int i = 0; i < query.vcount_; ++i)
    {
        int degree = query.vdegs_[i];
        position[i] = offset[degree];
        vertices[position[i]] = i;
        offset[degree] += 1;
    }

    for (int i = query.deg_max_; i > 0; --i)
    {
        offset[i] = offset[i - 1];
    }
    offset[0] = 0;

    for (int i = 0; i < query.vcount_; ++i)
    {
        int v = vertices[i];

        for (uint32_t offset1 = query.offsets_[v]; offset1 < query.offsets_[v + 1]; offset1 ++)
        {
            uint32_t u = query.neighbors_[offset1];

            if (core_table[u] > core_table[v]) {

                // Get the position and vertex which is with the same degree
                // and at the start position of vertices array.
                int cur_degree_u = core_table[u];
                int position_u = position[u];
                int position_w = offset[cur_degree_u];
                int w = vertices[position_w];

                if (u != w) {
                    // Swap u and w.
                    position[u] = position_w;
                    position[w] = position_u;
                    vertices[position_u] = w;
                    vertices[position_w] = u;
                }

                offset[cur_degree_u] += 1;
                core_table[u] -= 1;
            }
        }

        degeneracy_order[i] = v;
    }
}

void computeBFSOrder(
    const Graph& query, uint32_t *bfs_order, const uint32_t start_v
) {
    std::vector<uint32_t> pre_queue {start_v};
    std::vector<uint32_t> cur_queue;
    std::vector<bool> visited(query.vcount_, false);
    visited[start_v] = true;
    uint32_t pos = 0u;
    bfs_order[pos++] = start_v;

    while (!pre_queue.empty())
    {
        for (uint32_t u: pre_queue)
        {
            for (uint32_t offset = query.offsets_[u]; offset < query.offsets_[u + 1]; offset ++)
            {
                uint32_t uu = query.neighbors_[offset];
                if (!visited[uu])
                {
                    bfs_order[pos++] = uu;
                    cur_queue.push_back(uu);
                    visited[uu] = true;
                }
            }
        }
        std::swap(cur_queue, pre_queue);
        cur_queue.clear();
    }
}