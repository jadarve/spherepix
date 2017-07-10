/**
 * \file math.h
 * \brief Math types
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#ifndef SPHEREPIX_GPU_MATH_H_
#define SPHEREPIX_GPU_MATH_H_

namespace spherepix {
namespace gpu {

typedef struct {
    float3 row0;
    float3 row1;
} matrix23F_t;


typedef struct {
    float3 row0;
    float3 row1;
    float3 row2;
} matrix33F_t;


typedef struct {
    float4 row0;
    float4 row1;
    float4 row2;
    float4 row3;
} matrix44F_t;


}
}

#endif // SPHEREPIX_GPU_MATH_H_