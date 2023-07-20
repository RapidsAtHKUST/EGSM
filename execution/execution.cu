#include <chrono>

#include "utils/globals.h"
#include "utils/helpers.h"
#include "utils/cuda_helpers.h"
#include "utils/mem_pool.h"
#include "graph/graph.h"
#include "processing/plan.h"
#include "processing/join_bfs.h"
#include "structures/hashed_tries.h"
#include "structures/hashed_trie_manager.h"
#include "processing/join_bfs_dfs.h"


void copyConfig(
    const bool adaptive_ordering,
    const bool load_balancing
) {
    cudaErrorCheck(cudaMemcpyToSymbol(C_ADAPTIVE_ORDERING, &adaptive_ordering, sizeof(bool)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_LB_ENABLE, &load_balancing, sizeof(bool)));
}

void copyGraphMeta(
    const Graph& query_graph,
    const Graph& data_graph,
    const GraphUtils& query_utils
) {
    NUM_VQ = query_graph.vcount_;
    NUM_EQ = query_graph.ecount_;
    NUM_LQ = query_graph.lcount_;
    NUM_VD = data_graph.vcount_;
    MAX_L_FREQ = data_graph.lfreq_max_;
    memcpy(EIDX, query_utils.eidx_, sizeof(uint8_t) * NUM_VQ * NUM_VQ);
    memcpy(NBRBIT, query_utils.nbrbits_, sizeof(uint16_t) * NUM_VQ);
    cudaErrorCheck(cudaMemcpyToSymbol(C_NUM_VQ, &NUM_VQ, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_NUM_EQ, &NUM_EQ, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_NUM_LQ, &NUM_LQ, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_NUM_VD, &NUM_VD, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_MAX_L_FREQ, &MAX_L_FREQ, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_ORDER_OFFS, query_graph.offsets_, sizeof(uint32_t) * (NUM_VQ + 1)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_UTILS, &query_utils, sizeof(GraphUtils)));
    cudaErrorCheck(cudaDeviceSynchronize());
}

void copyTries(const HashedTries& hashed_tries)
{
    cudaErrorCheck(cudaMemcpyToSymbol(tries, &hashed_tries, sizeof(HashedTries)));
}

void matchDFSGroup(
    const HashedTrieManager& manager,
    Plan& plan,
    MemPool& pool,
    PoolElem& res,
    unsigned long long int& res_size
) {
    TIME_INIT();

    /*************** initialization ***************/
    // copy constant variables
    cudaErrorCheck(cudaMemcpyToSymbol(c_plan, &plan, sizeof(Plan)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_MEMPOOL, &pool, sizeof(MemPool)));

    // initialize temp variables
    uint8_t first_u = plan.start_vertex_id_;
    uint32_t first_min_off = manager.GetFirstMinOff(plan.start_vertex_id_);
    InitialOrder initial_order;
    memset(&initial_order, UINT8_MAX, sizeof(uint8_t) * MAX_VCOUNT);
    uint32_t *lb_triggered;
    cudaErrorCheck(cudaMalloc(&lb_triggered, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemset(lb_triggered, 0u, sizeof(uint32_t)));

    // initialize results
    PoolElem new_res = pool.TryMax();
    unsigned long long int *new_res_size;
    cudaErrorCheck(cudaMalloc(&new_res_size, sizeof(unsigned long long int)));
    cudaErrorCheck(cudaMemset(new_res_size, 0u, sizeof(unsigned long long int)));
    cudaErrorCheck(cudaDeviceSynchronize());

    uint32_t num_finished_groups = 0u;
    uint32_t num_mapped_vs = 1u, old_num_mapped_vs = 1u;
    uint32_t h_max_new_res_size = pool.GetFree();

    /*************** get the matches of the first vertex ***************/
    unsigned long long int num_warps = DIV_CEIL(manager.h_compacted_vs_sizes_[first_min_off], WARP_SIZE);
    uint32_t *pending_count;
    cudaErrorCheck(cudaMalloc(&pending_count, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemset(pending_count, 0u, sizeof(uint32_t)));

    TIME_START();
    joinDFSGroupKernel<<<DIV_CEIL(num_warps, WARP_PER_BLOCK), BLOCK_DIM>>>(
        res, num_warps,
        new_res, pool.capability_, new_res_size,
        initial_order, first_u, first_min_off,
        num_mapped_vs, true, pending_count, lb_triggered
    );
    cudaErrorCheck(cudaDeviceSynchronize());
    TIME_END();
    PRINT_LOCAL_TIME("Finish group " + std::to_string(num_finished_groups));

    unsigned long long int h_new_res_size = 0ul;
    cudaErrorCheck(cudaMemcpy(&h_new_res_size, new_res_size, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

    std::cout << "Group " << num_finished_groups << " # partial results: " << h_new_res_size << std::endl;
    pool.Push(h_new_res_size * num_mapped_vs);
    res = new_res;
    res_size = h_new_res_size;

    plan.AddGroup(num_finished_groups, initial_order);
    num_finished_groups ++;

    /*************** join a group at a time ***************/

    while (num_finished_groups < plan.mask_size_)
    {
        cudaErrorCheck(cudaMemset(new_res_size, 0u, sizeof(unsigned long long int)));
        cudaErrorCheck(cudaDeviceSynchronize());

        new_res = pool.TryMax();
        num_mapped_vs += popc(plan.masks_[num_finished_groups]);
        h_max_new_res_size = pool.GetFree() / num_mapped_vs;

        TIME_START();
        joinDFSGroupKernel<<<DIV_CEIL(res_size, WARP_PER_BLOCK), BLOCK_DIM>>>(
            res, res_size,
            new_res, h_max_new_res_size, new_res_size,
            initial_order, UINT8_MAX, UINT8_MAX,
            num_mapped_vs, true, pending_count, lb_triggered
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        TIME_END();

        cudaErrorCheck(cudaMemcpy(&h_new_res_size, new_res_size, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));


        if (h_new_res_size >= static_cast<unsigned long long int>(h_max_new_res_size))
        {
            PRINT_LOCAL_TIME("Stop group " + std::to_string(num_finished_groups));
            std::cout << "Combine remaining groups." << std::endl;
            break;
        }
        
        pool.Push(h_new_res_size * num_mapped_vs);
        pool.Pop(res_size * old_num_mapped_vs);
        res = new_res;
        res_size = h_new_res_size;

        plan.AddGroup(num_finished_groups, initial_order);
        PRINT_LOCAL_TIME("Finish group " + std::to_string(num_finished_groups));
        std::cout << "Group " << num_finished_groups << " # partial results: " << h_new_res_size << std::endl;

        num_finished_groups ++;
        old_num_mapped_vs = num_mapped_vs;
    }

    /*************** join all remaining query vertices ***************/
    if (num_finished_groups < plan.mask_size_)
    {
        cudaErrorCheck(cudaMemset(new_res_size, 0u, sizeof(unsigned long long int)));
        cudaErrorCheck(cudaDeviceSynchronize());

        TIME_START();
        joinDFSGroupKernel<<<DIV_CEIL(res_size, WARP_PER_BLOCK), BLOCK_DIM>>>(
            res, res_size,
            new_res, h_max_new_res_size, new_res_size,
            initial_order, UINT8_MAX, UINT8_MAX,
            NUM_VQ, false, pending_count, lb_triggered
        );
        cudaErrorCheck(cudaDeviceSynchronize());
        TIME_END();

        cudaErrorCheck(cudaMemcpy(&h_new_res_size, new_res_size, sizeof(unsigned long long int), cudaMemcpyDeviceToHost));

        PRINT_LOCAL_TIME("Finish last group");

        //pool.push(h_max_new_res_size * num_mapped_vs);
        //pool.pop(res_size * old_num_mapped_vs);
        //res = new_res;
        res_size = h_new_res_size;
    }
    std::cout << '\n';
    PRINT_TOTAL_TIME("Total join");

    uint32_t h_lb_triggered;
    cudaErrorCheck(cudaMemcpy(&h_lb_triggered, lb_triggered, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    std::cout << "Triggered Load Balancing: " << (h_lb_triggered > 0 ? "true" : "false") << '\n';

    cudaErrorCheck(cudaFree(lb_triggered));
    cudaErrorCheck(cudaFree(pending_count));
    cudaErrorCheck(cudaFree(new_res_size));
}
