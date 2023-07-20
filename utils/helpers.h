#ifndef UTILS_HELPERS_H
#define UTILS_HELPERS_H

#include <stdio.h>
#include <cstdint>


template<typename T>
uint32_t popc(T t)
{
    int count = 0;
    for (int i = 0; i < sizeof(T) * 8; i++)
    {
        if ((t >> i) & 1) count ++;
    }
    return count;
}

#endif
