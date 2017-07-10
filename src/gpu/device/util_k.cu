/**
 * \file util_k.h
 * \brief Device utility functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <flowfilter/gpu/device/image_k.h>

#include "spherepix/gpu/device/util_k.h"
#include "spherepix/gpu/device/math_k.h"

namespace spherepix {
namespace gpu {

using namespace flowfilter::gpu;

__global__ void float3ToFloat4Field(
    flowfilter::gpu::gpuimage_t<float3> f3Field,
    flowfilter::gpu::gpuimage_t<float4> f4Field) {


    const int height = f3Field.height;
    const int width = f3Field.width;

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= width || pix.y >= height) {
        return;
    }

    float3 v3 = *coordPitch(f3Field, pix);
    *coordPitch(f4Field, pix) = _make_float4(v3);
}


} // namespace gpu
} // namespace spherepix
