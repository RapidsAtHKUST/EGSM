# Efficient GPU-Accelerated Subgraph Matching

## Introduction

Subgraph matching is a basic operation in graph analytics, finding all occurrences of a query graph Q in a data graph G. A common approach is to first filter out non-candidate vertices in G, and then order the vertices in Q to enumerate results. Recent work has started to utilize the GPU to accelerate subgraph matching. However, the effectiveness of current GPU-based filtering and ordering methods is limited, and the result enumeration often runs out of memory quickly. To address these problems, we propose EGSM, an effcient approach to GPU-based subgraph matching. Speciffcally, we design a data structure Cuckoo trie to support dynamic maintenance of candidates for filtering, and order query vertices based on estimated numbers of candidate vertices on the fly. Furthermore, we perform a hybrid breadth-first and depth-first search with memory management for result enumeration. Consequently, EGSM significantly outperforms the state-of-the-art GPU-accelerated algorithms, including GSI and CuTS.

For the details, please refer to our SIGMOD'2023 paper "Efficient GPU-Accelerated Subgraph Matching" by [Xibo Sun](https://github.com/xibosun) and [Prof. Qiong Luo](https://cse.hkust.edu.hk/~luo/). If you have any further question, please feel free to contact us.

Please cite our paper if you use our source code.

- "Xibo Sun and Qiong Luo. Efficient GPU-Accelerated Subgraph Matching. SIGMOD 2023."

## Compile

Our program requires cmake (Version 3.21.3), Make (Version 3.82), GCC (Version 11.2.0), and nvcc (Version 11.7). One can compile the code by executing the following commands. 

```shell
mkdir build
cd build
cmake ..
make
cd ..
```

## Execute

After a successful compilation, the binary file is created under the `build/` directory. One can execute EGSM using the following command.

```shell
./build/EGSM -q <query-graph-path> -d <data-graph-path>
```

### Commandline Parameters
Other commandline parameters supported by the framework are listed in the following table.

| Parameters | Description                               | Valid Value     | Default Value |
|------------|-------------------------------------------|-----------------|---------------|
| -m         | Enumeration method.                       | BFS-DFS/BFS/DFS | BFS-DFS       |
| --f3       | Enable the third filtering step or not.   | on/off          | on            |
| --f3start  | Start vertex of the third filtering step. | 0-4294967295    | 4294967295    |
| --ao       | Enable adaptive ordering or not.          | on/off          | on            |
| --lb       | Enable load balancing or not.             | on/off          | on            |
| --gpu      | GPU ID for execution.                     | 0-4294967295    | 0             |

## Input File Format
Both the input query graph and data graph are vertex-labeled. Each graph starts with 't N M' where N is the number of vertices and M is the number of edges. Each vertex is represented by a distinct unsigned integer (from 0 to 4294967295). There is at most one edge between two arbitrary vertices. A vertex and an edge are formatted as `v <vertex-id> <vertex-label> <vertex-degree>` and `e <vertex-id-1> <vertex-id-2>`, respectively. The two endpoints of an edge must appear before the edge. For example, 

```
t 8 9
v 0 13 3
v 1 0 2
v 2 8 3
v 3 11 4
v 4 9 1
v 5 10 2
v 6 11 2
v 7 3 1
e 0 1
e 0 2
e 0 5
e 1 3
e 2 3
e 2 5
e 3 4
e 3 6
e 6 7
```

## Datasets and Querysets

The graph datasets and their corresponding querysets used in our paper can be downloaded [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/xsunax_connect_ust_hk/EqIAV7Qx6pdIvQWuMLnSpJMBgdwBIO_WlyjLeSlTxCaemw).
