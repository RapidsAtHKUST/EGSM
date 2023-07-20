#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <string>
#include <unordered_map>
#include <cuda_runtime.h>

#include "utils/CLI11.hpp"
#include "utils/config.h"
#include "utils/globals.h"
#include "utils/cuda_helpers.h"
#include "utils/mem_pool.h"

#include "graph/graph.h"
#include "structures/hashed_tries.h"
#include "structures/hashed_trie_manager.h"
#include "processing/plan.h"
#include "execution/execution.h"


int main(int argc, char** argv) {
    CLI::App app{"App description"};

    std::string query_path, data_path, method = "BFS-DFS";
    bool filtering_3rd = true, adaptive_ordering = true, load_balancing = true;
    uint32_t filtering_order_start_v = UINT32_MAX;
    uint32_t gpu_num = 0u;
    app.add_option("-q", query_path, "query graph path")->required();
    app.add_option("-d", data_path, "data graph path")->required();
    app.add_option("-m", method, "enumeration method");
    app.add_option("--f3", filtering_3rd, "enable the third filtering step or not");
    app.add_option("--f3start", filtering_order_start_v, "start vertex of the third filtering step");
    app.add_option("--ao", adaptive_ordering, "enable adaptive ordering or not");
    app.add_option("--lb", load_balancing, "enable load balancing or not");
    app.add_option("--gpu", gpu_num, "gpu number");

    CLI11_PARSE(app, argc, argv);

    cudaSetDevice(gpu_num);
    copyConfig(adaptive_ordering, load_balancing);

    /*************** read graph ***************/
    std::unordered_map<uint32_t, uint32_t> label_map;

    Graph query_graph(query_path, label_map);
    Graph data_graph(data_path, label_map);
    GraphGPU query_graph_gpu(query_graph);
    GraphGPU data_graph_gpu(data_graph);
    GraphUtils query_utils;
    query_utils.Set(query_graph);

    copyGraphMeta(query_graph, data_graph, query_utils);

    /*************** filtering ***************/
    HashedTries hashed_tries {};

    TIME_INIT();
    TIME_START();
    HashedTrieManager manager(query_graph, query_graph_gpu, data_graph_gpu, hashed_tries);
    TIME_END();
    PRINT_LOCAL_TIME("Build Cuckoo Tries");

    copyTries(hashed_tries);

    TIME_START();
    manager.Filter(hashed_tries, filtering_3rd, filtering_order_start_v);
    TIME_END();
    PRINT_LOCAL_TIME("Filtering");

    manager.GetCardinalities(hashed_tries);
    manager.Print();

    manager.Deallocate();
    query_graph_gpu.Deallocate();
    data_graph_gpu.Deallocate();

    Plan plan(query_graph, manager.h_compacted_vs_sizes_, method);

    plan.Print(query_graph);

    /*************** memory pool ***************/
    std::cout << '\n';
    MEM_INIT();
    PRINT_MEM_INFO("Before Allocation");

    MemPool pool {};
    pool.Alloc(MAX_RES_MEM_SPACE / sizeof(uint32_t));
    PoolElem res = pool.TryMax();
    unsigned long long int res_size = 0;

    PRINT_MEM_INFO("After allocation");
    std::cout << std::endl;;

    /*************** enumeration ***************/
    matchDFSGroup(manager, plan, pool, res, res_size);

    std::cout << "# Matches: " << res_size << "\nEnd." << std::endl;

    pool.Free();
    manager.DeallocateTries(hashed_tries);
}