/**
 * \file util.cu
 * \brief Utility functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <flowfilter/gpu/error.h>
#include <flowfilter/gpu/util.h>

#include "spherepix/gpu/util.h"

#include "spherepix/gpu/device/util_k.h"

namespace spherepix {
namespace gpu {

using namespace flowfilter::gpu;

void image3ToGPUImage4(spherepix::Image<float>& img3,
    flowfilter::gpu::gpuimage_t<float4> devImg4) {

    const int height = img3.height();
    const int width = img3.width();

    dim3 block(32, 32, 1);
    dim3 grid(0, 0, 0);
    configureKernelGrid(height, width, block, grid);


    // upload img3 to img3Dev
    flowfilter::image_t img3Wrapped = wrapToFlowFilterImage(img3);
    GPUImage img3Dev(height, width, 3, sizeof(float));
    img3Dev.upload(img3Wrapped);

    // transform input etas from float3 to float4 in device
    float3ToFloat4Field<<<grid, block>>>(
        img3Dev.wrap<float3>(), devImg4);

    checkError(cudaGetLastError());

}


} // namespace gpu
} // namespace spherepix
