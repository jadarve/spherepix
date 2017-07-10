/**
 * \file util_k.h
 * \brief Device utility functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef SPHEREPIX_GPU_UTIL_K_H_
#define SPHEREPIX_GPU_UTIL_K_H_

#include <flowfilter/gpu/image.h>


namespace spherepix {
namespace gpu {


__global__ void float3ToFloat4Field(
    flowfilter::gpu::gpuimage_t<float3> f3Field,
    flowfilter::gpu::gpuimage_t<float4> f4Field);


} // namespace gpu
} // namespace spherepix

#endif // SPHEREPIX_GPU_UTIL_K_H_