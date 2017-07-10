/**
 * \file pyramid_k.h
 * \brief Kernels for computing image pyramids.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#ifndef SPHEREPIX_GPU_PYRAMID_K_H_
#define SPHEREPIX_GPU_PYRAMID_K_H_

#include "spherepix/gpu/pixelation.h"

#include "spherepix/gpu/device/pixelation_k.h"

namespace spherepix {
namespace gpu {


template<typename T>
__global__ void imageDownB1_k(
    spherepix::gpu::pixfaceimg_t<T> inputImage,
    spherepix::gpu::pixfaceimg_t<T> imageDown) {


    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= imageDown.img.width || pix.y >= imageDown.img.height) {
        return;
    }

    // NOTE: the texture coordinates in B2 direction need to be multiplied
    //  by 2. This because input image width is double with respect to
    //  the width of the output imageDown

    T img_m = pixfaceimgRead(inputImage, make_int2(2*pix.x -1, pix.y));
    T img_0 = pixfaceimgRead(inputImage, make_int2(2*pix.x, pix.y));
    T img_p = pixfaceimgRead(inputImage, make_int2(2*pix.x +1, pix.y));

    float smoothed = 0.5*img_0 + 0.25*(img_m + img_p);

    // write output
    // pixfaceimgWrite(imageDown, pix, smoothed);
    pixfaceimgWrite(imageDown, pix, (T)(smoothed));
}


template<typename T>
__global__ void imageDownB2_k(
    spherepix::gpu::pixfaceimg_t<T> inputImage,
    spherepix::gpu::pixfaceimg_t<T> imageDown) {


    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= imageDown.img.width || pix.y >= imageDown.img.height) {
        return;
    }

    // NOTE: the texture coordinates in B1 direction need to be multiplied
    //  by 2. This because input image height is double with respect to
    //  the height of the output imageDown

    T img_m = pixfaceimgRead(inputImage, make_int2(pix.x, 2*pix.y -1));
    T img_0 = pixfaceimgRead(inputImage, make_int2(pix.x, 2*pix.y));
    T img_p = pixfaceimgRead(inputImage, make_int2(pix.x, 2*pix.y +1));

    float smoothed = 0.5*img_0 + 0.25*(img_m + img_p);


    // write output
    // pixfaceimgWrite(imageDown, pix, smoothed);
    pixfaceimgWrite(imageDown, pix, (T)smoothed);
}


} // namespace gpu
} // namepsace spherepix

#endif /* SPHEREPIX_GPU_PYRAMID_K_H_ */