cmake_minimum_required(VERSION 3.10)
project(EGSM LANGUAGES CUDA CXX)
set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE RELEASE)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(/usr/local/cuda-11.7/include)

add_library(nd SHARED 
    utils/nucleus/bucket.cpp
    utils/nucleus/hierarchy.cpp utils/nucleus/k3-4.cpp
    utils/nucleus/kcore.cpp utils/nucleus/ktruss.cpp
    utils/nucleus/nd_interface.cpp utils/nucleus/util.cpp
    utils/nucleus/bucket.h utils/nucleus/nd_interface.h
    utils/nucleus/nd.h)

set_target_properties(nd PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

add_library(link STATIC
    utils/CLI11.hpp utils/config.h
    utils/cuda_helpers.h utils/helpers.h
    utils/globals.h utils/globals.cu
    utils/mem_pool.h utils/mem_pool.cu
    utils/search.cuh utils/types.h
    
    structures/hashed_trie_manager_kernel.h
    structures/hashed_trie_manager_kernel.cu
    structures/hashed_trie_manager.h
    structures/hashed_trie_manager.cu
    structures/hashed_tries.h
    structures/hashed_tries.cu
    
    processing/common.h
    processing/join_bfs_dfs.h
    processing/join_bfs_dfs.cu
    processing/join_bfs.h
    processing/join_bfs.cu
    processing/plan.h
    processing/plan.cu
    
    graph/graph_gpu.h
    graph/graph_gpu.cu
    
    execution/execution.h execution/execution.cu)

set_target_properties(link PROPERTIES
    CUDA_ARCHITECTURES "70"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(link PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    -Xptxas=\"-v\"
    -DTHRUST_IGNORE_CUB_VERSION_CHECK
    >)


add_executable(EGSM
    main.cpp
    graph/graph.h
    graph/graph.cpp
    graph/operations.h
    graph/operations.cpp)

set_target_properties(EGSM PROPERTIES
    CUDA_ARCHITECTURES "70"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON)

target_compile_options(EGSM PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    --generate-line-info
    -Xptxas=\"-v\"
    -DTHRUST_IGNORE_CUB_VERSION_CHECK
    >)

target_link_libraries(EGSM PUBLIC link nd)
