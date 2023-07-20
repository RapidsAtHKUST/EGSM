#ifndef UTILS_TYPES_H
#define UTILS_TYPES_H

#include <cstdint>
#include <cub/cub.cuh>

#include "utils/config.h"


struct Bucket
{
    uint32_t data[BUCKET_DIM];
};

struct NotUintMax
{
    CUB_RUNTIME_FUNCTION __forceinline__
    NotUintMax() {}
    CUB_RUNTIME_FUNCTION __forceinline__
    bool operator()(const uint32_t &a) const {
        return (a != UINT32_MAX);
    }
};


#endif //UTILS_TYPES_H
