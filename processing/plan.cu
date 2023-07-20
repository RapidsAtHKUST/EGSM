#include <algorithm>
#include <string>
#include <queue>
#include <unordered_set>
#include <vector>

#include "utils/globals.h"
#include "utils/helpers.h"
#include "graph/graph.h"
#include "utils/nucleus/nd_interface.h"
#include "processing/plan.h"

__constant__ Plan c_plan;

void remove_tree_vs(
    const Graph& query,
    nd_Graph& core,
    uint16_t& tree_vs_map
) {
    nd_interface::convert_graph(query, core);

    // get all vertices in trees
    bool any_vertex_removed = true;
    while (any_vertex_removed)
    {
        any_vertex_removed = false;
        for (uint32_t i = 0u; i < core.size(); i++)
        {
            if (core[i].size() == 1)
            {
                uint32_t u_other = core[i][0];
                core[i].clear();
                auto it = std::lower_bound(
                    core[u_other].begin(),
                    core[u_other].end(),
                    i
                );
                if (it == core[u_other].end() || *it != i)
                {
                    std::cout << "cannot find vertex in adjacent list!\n";
                    exit(-1);
                }
                core[u_other].erase(it);

                tree_vs_map |= (1 << i);
                any_vertex_removed = true;
            }
        }
    }
}

void extract_densest_core_from_nucleus(
    const Graph& query,
    std::vector<nd_tree_node>& k34_tree,
    uint16_t& densest_core_map
) {
    // 1. get exclusive nucleus
    // each item is a vertex group
    std::vector<std::vector<uint32_t>> exclusive_nucleus;
    // each item contains all vertices in the group and all neighbors of each vertex
    std::vector<std::vector<uint32_t>> greater_exclusive_nucleus;
    // check the root of each k34_tree
    for (auto node: k34_tree)
    {
        uint32_t merge_idx = UINT32_MAX;

        for (uint32_t i = 0u; i < exclusive_nucleus.size(); i++)
        {
            auto& nbrs = greater_exclusive_nucleus[i];
            std::vector<uint32_t> result(std::min(node.vertices_.size(), nbrs.size()));
            if (std::set_intersection(
                node.vertices_.begin(), node.vertices_.end(),
                nbrs.begin(), nbrs.end(),
                result.begin()) != result.begin()
            ) {
                // the new node have some vertex in common with greater_exclusive_nucleus[i]
                merge_idx = i;
                break;
            }
        }
        if (merge_idx == UINT32_MAX)
        {
            // create a new group
            merge_idx = exclusive_nucleus.size();
            exclusive_nucleus.emplace_back();
            greater_exclusive_nucleus.emplace_back();
        }

        // merge all the vertices in node to an existing group
        std::vector<uint32_t> new_core(exclusive_nucleus[merge_idx].size() + node.vertices_.size());
        auto it = std::set_union(
            exclusive_nucleus[merge_idx].begin(), exclusive_nucleus[merge_idx].end(),
            node.vertices_.begin(), node.vertices_.end(),
            new_core.begin()
        );
        new_core.resize(it - new_core.begin());
        std::swap(new_core, exclusive_nucleus[merge_idx]);

        // merge all the neighbors of each vertex in node to an existing group
        for (auto v: node.vertices_)
        {
            std::vector<uint32_t> nbrs(query.neighbors_ + query.offsets_[v], query.neighbors_ + query.offsets_[v + 1]);
            std::vector<uint32_t> new_greater_core(greater_exclusive_nucleus[merge_idx].size() + nbrs.size());
            auto it = std::set_union(
                greater_exclusive_nucleus[merge_idx].begin(), greater_exclusive_nucleus[merge_idx].end(),
                nbrs.begin(), nbrs.end(),
                new_greater_core.begin()
            );
            new_greater_core.resize(it - new_greater_core.begin());
            std::swap(greater_exclusive_nucleus[merge_idx], new_greater_core);
        }
    }
    // 2. find the nucleus with maximum number of vertices
    uint32_t selected_idx = 0u;
    if (exclusive_nucleus.size() > 1)
    {
        uint32_t max_size = 0u;
        for (uint32_t i = 0u; i < exclusive_nucleus.size(); i++)
        {
            if (exclusive_nucleus[i].size() > max_size)
            {
                max_size = exclusive_nucleus[i].size();
                selected_idx = i;
            }
        }
    }
    for (auto v: exclusive_nucleus[selected_idx])
    {
        densest_core_map |= (1 << v);
    }
}

void extract_densest_core_from_core(
    const Graph& query,
    const nd_Graph core,
    uint16_t& densest_core_map
) {
    // first try to find a triangle
    std::vector<std::vector<uint32_t>> triangles;
    for (uint32_t v = 0u; v < core.size(); v++)
    {
        if (core[v].empty()) continue;
        for (uint32_t v_other: core[v])
        {
            if (v_other <= v) continue;
            std::vector<uint32_t> result(core.size());
            auto end = std::set_intersection(
                core[v].begin(), core[v].end(),
                core[v_other].begin(), core[v_other].end(),
                result.begin()
            );
            for (auto it = result.begin(); it != end; it++)
            {
                if (*it > v_other)
                {
                    triangles.push_back({v, v_other, *it});
                }
            }
        }
    }
    if (!triangles.empty())
    {
        // if there is at least one triangle, find one
        uint32_t selected_idx, max_degree = 0u;
        for (uint32_t i = 0u; i < triangles.size(); i++)
        {
            const auto& triangle = triangles[i];
            const uint32_t local_degree = query.vdegs_[triangle[0]] + query.vdegs_[triangle[1]] + query.vdegs_[triangle[2]];
            if (local_degree > max_degree)
            {
                selected_idx = i;
                max_degree = local_degree;
            }
        }
        densest_core_map |= (1 << triangles[selected_idx][0]);
        densest_core_map |= (1 << triangles[selected_idx][1]);
        densest_core_map |= (1 << triangles[selected_idx][2]);
    }
    else
    {
        // if there is no triangle, find an edge
        uint32_t selected_v, selected_v_other, max_degree = 0u;
        for (uint32_t v = 0u; v < core.size(); v++)
        {
            if (core[v].empty()) continue;
            for (auto v_other: core[v])
            {
                if (v_other <= v) continue;
                const uint32_t local_degree = query.vdegs_[v] + query.vdegs_[v_other];
                if (local_degree > max_degree)
                {
                    max_degree = local_degree;
                    selected_v = v;
                    selected_v_other = v_other;
                }
            }
        }
        densest_core_map |= (1 << selected_v);
        densest_core_map |= (1 << selected_v_other);
    }
}

void extract_densest_core_from_tree(
    const Graph& query,
    uint16_t& densest_core_map
) {
    // find an edge from the tree
    uint32_t selected_v, selected_v_other, max_degree = 0u;
    for (uint32_t v = 0u; v < query.vcount_; v++)
    {
        for (uint32_t off = query.offsets_[v]; off < query.offsets_[v + 1]; off++)
        {
            const uint32_t v_other = query.neighbors_[off];
            if (v_other <= v) continue;
            const uint32_t local_degree = query.vdegs_[v] + query.vdegs_[v_other];
            if (local_degree > max_degree)
            {
                max_degree = local_degree;
                selected_v = v;
                selected_v_other = v_other;
            }
        }
    }
    densest_core_map |= (1 << selected_v);
    densest_core_map |= (1 << selected_v_other);
}


Plan::Plan(
    const Graph& query,
    uint32_t *relation_sizes,
    std::string method
) {
    // get all outer trees of the query graph
    nd_Graph core;

    uint16_t tree_vs_map = 0;
    uint16_t core_vs_map = 0;
    uint16_t densest_core_map = 0;
    uint16_t remain_core_map = 0;

    if (query.vcount_ - 1 < query.ecount_)
    {
        remove_tree_vs(query, core, tree_vs_map);
        core_vs_map = (~tree_vs_map & ~(0xffff << query.vcount_));
    }
    else
    {
        tree_vs_map = pow(2, query.vcount_) - 1;
    }

    std::vector<nd_tree_node> k34_tree;
    nd_interface::nd(core, 3, 4, k34_tree);

    if (!k34_tree.empty())
    {
        // the query contains a nuclei
        extract_densest_core_from_nucleus(
            query,
            k34_tree,
            densest_core_map
        );
    }
    else
    {
        if (core_vs_map)
        {
            // the query is not a tree
            extract_densest_core_from_core(
                query, core, densest_core_map
            );
        }
        else
        {
            // the query is a tree
            extract_densest_core_from_tree(
                query, densest_core_map
            );
            tree_vs_map &= (~densest_core_map);
            core_vs_map = densest_core_map;
        }
    }
    
    remain_core_map = (core_vs_map & ~densest_core_map);
    std::vector<std::vector<uint32_t>> sparses;

    if (remain_core_map)
    {
        // still have remaining core vertices, then put remaining vertices into groups
        std::vector<bool> visited(query.vcount_, false);
        for (uint32_t i = 0u; i < query.vcount_; i++)
        {
            if (densest_core_map & (1 << i) == 0)
                continue;
            for (auto v: core[i])
            {
                if (!(!visited[v] && remain_core_map & (1 << v)))
                    continue;

                // start a new group
                visited[v] = true;

                sparses.emplace_back();
                sparses.back().push_back(v);

                std::vector<int> cur_level{v};
                std::vector<int> next_level;

                while (!cur_level.empty())
                {
                    for (int cur_v: cur_level)
                    {
                        for (int v_other: core[cur_v])
                        {
                            if (!(!visited[v_other] && (remain_core_map & (1 << v_other))))
                                continue;

                            next_level.push_back(v_other);
                            visited[v_other] = true;
                        }
                    }
                    cur_level.clear();
                    for (auto v_other: next_level)
                    {
                        sparses.back().push_back(v_other);
                    }
                    std::swap(cur_level, next_level);
                }
            }
        }
    }
    
    // put remaining tree vertices into groups
    std::vector<std::vector<uint32_t>> trees;
    std::vector<bool> visited(query.vcount_, false);
    for (uint32_t i = 0u; i < query.vcount_; i++)
    {
        if (tree_vs_map & (1 << i))
            continue;
        for (uint32_t off = query.offsets_[i]; off < query.offsets_[i + 1]; off++)
        {
            int v = query.neighbors_[off];

            if (!(!visited[v] && (tree_vs_map & (1 << v))))
                continue;

            // start a new group
            visited[v] = true;

            trees.emplace_back();
            trees.back().push_back(v);

            std::vector<int> cur_level{v};
            std::vector<int> next_level;

            while (!cur_level.empty())
            {
                for (int cur_v: cur_level)
                {
                    for (uint32_t off = query.offsets_[cur_v]; off < query.offsets_[cur_v + 1]; off++)
                    {
                        int v_other = query.neighbors_[off];

                        if (!(!visited[v_other] && (tree_vs_map & (1 << v_other))))
                            continue;

                        next_level.push_back(v_other);
                        visited[v_other] = true;
                    }
                }
                cur_level.clear();
                for (auto v_other: next_level)
                {
                    trees.back().push_back(v_other);
                }
                std::swap(cur_level, next_level);
            }
        }
    }
    
    // order all groups
    std::sort(
        sparses.begin(), sparses.end(),
        [&query, &densest_core_map, &relation_sizes]
        (const auto& s1, const auto& s2)
        {
            uint32_t num_backwards1 = 0u;
            uint32_t min_num_candidates1 = UINT32_MAX;
            uint32_t num_backwards2 = 0u;
            uint32_t min_num_candidates2 = UINT32_MAX;
            for (auto v_s1: s1)
            {
                for (uint32_t off = query.offsets_[v_s1]; off < query.offsets_[v_s1 + 1]; off++)
                {
                    const auto v_other = query.neighbors_[off];
                    if (densest_core_map && (1 << v_other))
                    {
                        if (relation_sizes[EIDX[v_s1 * query.vcount_ + v_other]] < min_num_candidates1)
                        {
                            min_num_candidates1 = relation_sizes[EIDX[v_s1 * query.vcount_ + v_other]];
                        }
                        num_backwards1++;
                    }
                }
            }
            for (auto v_s2: s2)
            {
                for (uint32_t off = query.offsets_[v_s2]; off < query.offsets_[v_s2 + 1]; off++)
                {
                    const auto v_other = query.neighbors_[off];
                    if (densest_core_map && (1 << v_other))
                    {
                        if (relation_sizes[EIDX[v_s2 * query.vcount_ + v_other]] < min_num_candidates2)
                        {
                            min_num_candidates2 = relation_sizes[EIDX[v_s2 * query.vcount_ + v_other]];
                        }
                        num_backwards2++;
                    }
                }
            }
            return num_backwards1 > num_backwards2 || (
                num_backwards1 == num_backwards2 &&
                min_num_candidates1 < min_num_candidates2
            );
        }
    );

    std::sort(
        trees.begin(), trees.end(),
        [&query, &core_vs_map, &relation_sizes]
        (const auto& s1, const auto& s2)
        {
            uint32_t min_num_candidates1 = UINT32_MAX;
            uint32_t size1 = UINT32_MAX;
            uint32_t min_num_candidates2 = UINT32_MAX;
            uint32_t size2 = UINT32_MAX;
            for (auto v_s1: s1)
            {
                for (uint32_t off = query.offsets_[v_s1]; off < query.offsets_[v_s1 + 1]; off++)
                {
                    const auto v_other = query.neighbors_[off];
                    if (core_vs_map && (1 << v_other))
                    {
                        if (relation_sizes[EIDX[v_s1 * query.vcount_ + v_other]] < min_num_candidates1)
                        {
                            min_num_candidates1 = relation_sizes[EIDX[v_s1 * query.vcount_ + v_other]];
                        }
                    }
                }
                size1++;
            }
            for (auto v_s2: s2)
            {
                for (uint32_t off = query.offsets_[v_s2]; off < query.offsets_[v_s2 + 1]; off++)
                {
                    const auto v_other = query.neighbors_[off];
                    if (core_vs_map && (1 << v_other))
                    {
                        if (relation_sizes[EIDX[v_s2 * query.vcount_ + v_other]] < min_num_candidates2)
                        {
                            min_num_candidates2 = relation_sizes[EIDX[v_s2 * query.vcount_ + v_other]];
                        }
                    }
                }
                size2++;
            }
            return size1 < size2 || (
                size1 == size2 &&
                min_num_candidates1 < min_num_candidates2
            );
        }
    );

    // find the first vertex
    uint32_t first_u_degree = 0u;
    for (uint32_t i = 0; i < query.vcount_; i++)
    {
        if (densest_core_map & (1 << i) && query.vdegs_[i] > first_u_degree)
        {
            first_u_degree = query.vdegs_[i];
            start_vertex_id_ = i;
        }
    }

    uint32_t pos = 0u;
    is_tree_group_ = 0u;
    for (uint32_t i = 0; i < MAX_VCOUNT; i++)
        masks_[i] = 0u;
    masks_[pos] = 1 << start_vertex_id_;
    mask_size_prefix_sum_[pos++] = 0;

    if (method == "BFS-DFS")
    {
        densest_core_map &= (~masks_[0]);
        masks_[pos] = densest_core_map;
        mask_size_prefix_sum_[pos++] = 1;
        for (uint32_t i = 0; i < sparses.size(); i++)
        {
            uint16_t local_mask = 0u;
            for (auto v: sparses[i])
            {
                local_mask |= (1 << v);
            }
            masks_[pos] = local_mask;
            mask_size_prefix_sum_[pos] = popc(masks_[pos - 1]) + mask_size_prefix_sum_[pos - 1];
            pos++;
        }
        for (uint32_t i = 0; i < trees.size(); i++)
        {
            uint16_t local_mask = 0u;
            for (auto v: trees[i])
            {
                local_mask |= (1 << v);
            }
            masks_[pos] = local_mask;
            is_tree_group_ |= (1 << pos);
            mask_size_prefix_sum_[pos] = popc(masks_[pos - 1]) + mask_size_prefix_sum_[pos - 1];
            pos++;
        }
    }
    else if (method == "BFS")
    {
        uint16_t local_visited = (1 << start_vertex_id_);
        for (uint32_t i = 0; i < popc(densest_core_map) - 1; i++)
        {
            uint32_t min_vsize = UINT32_MAX, next_u = 0;
            for (uint32_t j = 0; j < query.vcount_; j++)
            {
                if ((densest_core_map & (1 << j)) && (NBRBIT[j] & local_visited) && ((1 << j) & ~local_visited) && (relation_sizes[EIDX[j * query.vcount_ + query.neighbors_[query.offsets_[j]]]] < min_vsize))
                {
                    min_vsize = relation_sizes[EIDX[j * query.vcount_ + query.neighbors_[query.offsets_[j]]]];
                    next_u = j;
                }
            }
            masks_[pos] = 1 << next_u;
            mask_size_prefix_sum_[pos] = mask_size_prefix_sum_[pos - 1] + 1;
            local_visited |= (1 << next_u);
            pos++;
        }
        for (auto& sparse: sparses)
        {
            for (uint32_t i = 0; i < sparse.size(); i++)
            {
                uint32_t min_vsize = UINT32_MAX, next_u = 0;
                for (auto j: sparse)
                {
                    if ((NBRBIT[j] & local_visited) && ((1 << j) & ~local_visited) && (relation_sizes[EIDX[j * query.vcount_ + query.neighbors_[query.offsets_[j]]]] < min_vsize))
                    {
                        min_vsize = relation_sizes[EIDX[j * query.vcount_ + query.neighbors_[query.offsets_[j]]]];
                        next_u = j;
                    }
                }
                masks_[pos] = 1 << next_u;
                mask_size_prefix_sum_[pos] = mask_size_prefix_sum_[pos - 1] + 1;
                local_visited |= (1 << next_u);
                pos++;
            }
        }
        for (auto& tree: trees)
        {
            for (uint32_t i = 0; i < tree.size(); i++)
            {
                uint32_t min_vsize = UINT32_MAX, next_u = 0;
                for (auto j: tree)
                {
                    if ((NBRBIT[j] & local_visited) && ((1 << j) & ~local_visited) && (relation_sizes[EIDX[j * query.vcount_ + query.neighbors_[query.offsets_[j]]]] < min_vsize))
                    {
                        min_vsize = relation_sizes[EIDX[j * query.vcount_ + query.neighbors_[query.offsets_[j]]]];
                        next_u = j;
                    }
                }
                masks_[pos] = 1 << next_u;
                mask_size_prefix_sum_[pos] = mask_size_prefix_sum_[pos - 1] + 1;
                local_visited |= (1 << next_u);
                pos++;
            }
        }
    }
    else
    {
        mask_size_prefix_sum_[pos] = 1;
        for (uint32_t i = 0; i < query.vcount_; i++)
        {
            if (i != start_vertex_id_)
            {
                masks_[pos] |= 1 << i;
            }
        }
        pos++;
    }
    mask_size_ = pos;

    pos = 0u;
    for (uint32_t i = 0; i < mask_size_; i++)
    {
        for (uint8_t j = 0; j < 32; j++)
        {
            if ((masks_[i] >> j) & 1)
            {
                res_pos[j] = pos;
                pos++;
            }
        }
    }
}


Plan::Plan(
    const Graph& query,
    uint32_t *relation_sizes,
    uint32_t gsi
) {
    mask_size_ = query.vcount_;
    std::vector<bool> visited(query.vcount_);
    std::vector<bool> selected(query.vcount_);

    auto comp = [](const auto& v1, const auto& v2){
        return v1.second < v2.second;
    };
    std::priority_queue<
        std::pair<uint32_t, uint32_t>,
        std::vector<std::pair<uint32_t, uint32_t>>,
        decltype(comp)> extendable_vertices(comp);

    // get the query vertex with largest degree
    uint32_t first_u_degree = 0u, pos = 0u;
    for (uint32_t i = 0; i < query.vcount_; i++)
    {
        if (query.vdegs_[i] > first_u_degree)
        {
            first_u_degree = query.vdegs_[i];
            start_vertex_id_ = i;
        }
    }
    mask_size_prefix_sum_[pos] = 0;
    masks_[pos++] = 1 << start_vertex_id_;
    visited[start_vertex_id_] = true;
    selected[start_vertex_id_] = true;

    for (uint32_t off = query.offsets_[start_vertex_id_]; off < query.offsets_[start_vertex_id_ + 1]; off++)
    {
        const auto nbr = query.neighbors_[off];
        visited[nbr] = true;
        extendable_vertices.emplace(nbr, query.vdegs_[nbr]);
    }
    while (!extendable_vertices.empty())
    {
        const uint32_t selected_v = extendable_vertices.top().first;
        extendable_vertices.pop();

        selected[selected_v] = true;
        mask_size_prefix_sum_[pos] = mask_size_prefix_sum_[pos - 1] + 1;
        masks_[pos] = 1 << selected_v;
        if (query.vdegs_[selected_v] == 1)
        {
            is_tree_group_ |= (1 << pos);
        }
        pos++;

        for (uint32_t off = query.offsets_[selected_v]; off < query.offsets_[selected_v + 1]; off++)
        {
            const auto nbr = query.neighbors_[off];
            if (visited[nbr]) continue;
            extendable_vertices.emplace(nbr, query.vdegs_[nbr]);
            visited[nbr] = true;
        }
    }
    mask_size_ = pos;
    
    pos = 0u;
    for (uint32_t i = 0; i < mask_size_; i++)
    {
        for (uint8_t j = 0; j < 32; j++)
        {
            if ((masks_[i] >> j) & 1)
            {
                res_pos[j] = pos;
                pos++;
            }
        }
    }
}


void Plan::AddGroup(uint8_t i, InitialOrder& initial_order)
{
    uint8_t cur_pos = mask_size_prefix_sum_[i];
    for (uint8_t j = 0; j < 32; j++)
    {
        if ((masks_[i] >> j) & 1)
        {
            initial_order.u[cur_pos++] = j;
        }
    }
}

void Plan::Print(const Graph& query)
{
    std::vector<bool> visited(NUM_VQ, false);
    for (uint32_t i = 0; i < mask_size_; i++)
    {
        std::cout << "Group " << i << ": ";
        std::vector<uint32_t> this_group;
        std::vector<bool> this_group_bitmap(NUM_VQ, false);
        for (uint32_t j = 0; j < 32; j++)
        {
            if ((masks_[i] >> j) & 1)
            {
                this_group.push_back(j);
                this_group_bitmap[j] = true;
                std::cout << j << ' ';
            }
        }
        if ((1 << i) & is_tree_group_)
        {
            std::cout << "(tree)";
        }

        std::cout << "\nGroup " << i << " pre edges: ";
        for (uint32_t v: this_group)
        {
            for (uint32_t off = query.offsets_[v]; off < query.offsets_[v + 1]; off++)
            {
                const uint32_t v_other = query.neighbors_[off];
                if (visited[v_other])
                {
                    std::cout << v_other << " - " << v << ' ';
                }
            }
        }

        std::cout << "\nGroup " << i << " cur edges: ";
        for (uint32_t v: this_group)
        {
            for (uint32_t off = query.offsets_[v]; off < query.offsets_[v + 1]; off++)
            {
                const uint32_t v_other = query.neighbors_[off];
                if (this_group_bitmap[v_other])
                {
                    if (v < v_other) continue;
                    std::cout << v_other << " - " << v << ' ';
                }
            }
            visited[v] = true;
        }
        std::cout << '\n';
    }
}
