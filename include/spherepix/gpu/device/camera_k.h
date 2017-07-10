/**
 * \file camera_k.h
 * \brief Device functions for camera.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef SPHEREPIX_GPU_CAMERA_K_H_
#define SPHEREPIX_GPU_CAMERA_K_H_

#include "spherepix/gpu/pixelation.h"

#include "spherepix/gpu/device/pixelation_k.h"

namespace spherepix {
namespace gpu {


// NOTE:
// There is a problem doing image mapping for uint8 pixel type, the
// returned image is zeros.
template<typename T>
__global__ void interpolateImage_k(
    spherepix::gpu::pixfaceimg_t<T> inputImage,
    spherepix::gpu::pixfaceimg_t<float2> faceBetas,
    spherepix::gpu::pixfaceimg_t<T> outputImage,
    T fillValue) {

    // pixel coordinate
    const int2 pix = make_int2(blockIdx.x*blockDim.x + threadIdx.x,
    blockIdx.y*blockDim.y + threadIdx.y);

    if(pix.x >= faceBetas.img.width || pix.y >= faceBetas.img.height) {
        return;
    }

    // input image size
    const float2 iSize = make_float2(inputImage.img.width, inputImage.img.height);

    // read beta coordinate (col, row)
    // float2 beta = swap(pixfaceimgRead(faceBetas, pix));
    float2 beta = pixfaceimgRead(faceBetas, pix);

    // read from input image or set to zero if out of range
    T imgValue = inRange(beta, iSize)? pixfaceimgRead(inputImage, beta) : fillValue;

    // write output
    pixfaceimgWrite(outputImage, pix, imgValue);


    // this approach is 0.01 ms faster, not much worth it
    // considering it does not fill the other parts of the image.
    // Reconsider using it for the case of multiple cameras mapping
    // to the same face.

    // if(inRange(beta, iSize)) {
    //     T imgValue = pixfaceimgRead(inputImage, beta);
    //     // write output
    //     pixfaceimgWrite(outputImage, pix, imgValue);        
    // }
}

} // namespace gpu
} // namespace spherepix

#endif // SPHEREPIX_GPU_CAMERA_K_H_