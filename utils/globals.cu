#ifndef UTILS_GLOBALS_H
#define UTILS_GLOBALS_H


#include <cstdint>

#include "utils/config.h"
#include "graph/graph.h"


uint32_t NUM_VQ;
uint32_t NUM_EQ;
uint32_t NUM_LQ;
uint32_t NUM_VD;
uint32_t MAX_L_FREQ;
uint8_t EIDX[MAX_VCOUNT * MAX_VCOUNT];
uint16_t NBRBIT[MAX_VCOUNT];

__constant__ bool C_ADAPTIVE_ORDERING;
__constant__ bool C_LB_ENABLE;

__constant__ uint32_t C_NUM_VQ;
__constant__ uint32_t C_NUM_EQ;
__constant__ uint32_t C_NUM_LQ;
__constant__ uint32_t C_NUM_VD;
__constant__ uint32_t C_MAX_L_FREQ;

__constant__ GraphUtils C_UTILS;

__constant__ uint32_t C_ORDER[MAX_VCOUNT];
__constant__ uint32_t C_ORDER_OFFS[MAX_VCOUNT + 1];

__constant__ uint32_t C[MAX_ECOUNT * 2 * 2 * 2];

__constant__ uint32_t C_TABLE_OFFS[MAX_ECOUNT * 2];
__constant__ uint32_t C_NUM_BUCKETS[MAX_ECOUNT * 2];


#endif //UTILS_GLOBALS_H
