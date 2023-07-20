#ifndef UTILS_GLOBALS_H
#define UTILS_GLOBALS_H


#include <cstdint>

#include "utils/config.h"
#include "graph/graph.h"


extern uint32_t NUM_VQ;
extern uint32_t NUM_EQ;
extern uint32_t NUM_LQ;
extern uint32_t NUM_VD;
extern uint32_t MAX_L_FREQ;
// edge index given the two endpoints
extern uint8_t EIDX[MAX_VCOUNT * MAX_VCOUNT];
// the i-th bit is 1 in C_NBRBIT[j] if there is an edge between i and j
extern uint16_t NBRBIT[MAX_VCOUNT];

extern __constant__ bool C_ADAPTIVE_ORDERING;
extern __constant__ bool C_LB_ENABLE;

extern __constant__ uint32_t C_NUM_VQ;
extern __constant__ uint32_t C_NUM_EQ;
extern __constant__ uint32_t C_NUM_LQ;
extern __constant__ uint32_t C_NUM_VD;
extern __constant__ uint32_t C_MAX_L_FREQ;

extern __constant__ GraphUtils C_UTILS;

extern __constant__ uint32_t C_ORDER[MAX_VCOUNT];
extern __constant__ uint32_t C_ORDER_OFFS[MAX_VCOUNT + 1];

// constants C0 and C1 for each hash table
extern __constant__ uint32_t C[MAX_ECOUNT * 2 * 2 * 2];

extern __constant__ uint32_t C_TABLE_OFFS[MAX_ECOUNT * 2];
extern __constant__ uint32_t C_NUM_BUCKETS[MAX_ECOUNT * 2];


#endif //UTILS_GLOBALS_H
