/**
 * \file util.h
 * \brief Utility functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef SPHEREPIX_GPU_UTIL_H_
#define SPHEREPIX_GPU_UTIL_H_

#include <flowfilter/gpu/image.h>

#include "spherepix/image.h"

namespace spherepix {
namespace gpu {


template<typename T>
flowfilter::image_t wrapToFlowFilterImage(spherepix::Image<T>& img) {

    flowfilter::image_t imgw;
    imgw.height = img.height();
    imgw.width = img.width();
    imgw.depth = img.depth();
    imgw.pitch = img.pitch();
    imgw.itemSize = sizeof(T);
    imgw.data = img.data();

    return imgw;
}


void image3ToGPUImage4(spherepix::Image<float>& img3,
    flowfilter::gpu::gpuimage_t<float4> devImg4);


} // namespace gpu
} // namespace spherepix

#endif // SPHEREPIX_GPU_UTIL_H_