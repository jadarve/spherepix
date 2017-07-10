/**
 * \file camera.cu
 * \brief GPU camera classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <exception>

#include <flowfilter/gpu/util.h>

#include "spherepix/pixelation.h"
#include "spherepix/gpu/camera.h"
#include "spherepix/gpu/util.h"

#include "spherepix/gpu/device/camera_k.h"

namespace spherepix {
namespace gpu {

using namespace spherepix;


//###############################################
// GPUCamera
//###############################################

GPUCamera::GPUCamera() {

}


GPUCamera::GPUCamera(std::shared_ptr<spherepix::Camera> hostCamera) {
    __hostCamera = hostCamera;
    configure();
}


GPUCamera::~GPUCamera() {
    // nothing to do
}


void GPUCamera::configure() {

    std::cout << "GPUCamera::configure(): start" << std::endl;

    Image<float> camEtas = __hostCamera->sphericalCoordinates();

    // GPU spherical coordinates
    __etas = PixelationFaceImage(camEtas.height(),
        camEtas.width(), 4, FLOAT32);

    image3ToGPUImage4(camEtas, __etas.wrapImage<float4>());

    std::cout << "GPUCamera::configure(): finished" << std::endl;
}

std::shared_ptr<spherepix::Camera> GPUCamera::getHostCamera() {
    return __hostCamera;
}

int GPUCamera::height() const {
    return __hostCamera->height();
}

int GPUCamera::width() const {
    return __hostCamera->width();
}


//###############################################
// GPUFaceImageMapper
//###############################################
GPUFaceImageMapper::GPUFaceImageMapper() {
    __configured = false;
}


GPUFaceImageMapper::GPUFaceImageMapper(GPUCamera camera, PixelationFace face) {

    __configured = false;
    __useCameraOrFaceBetas = true;
    __camera = camera;
    __face = face;
    // cannot call configure() until inputImage has been set
    // configure(); 
}

GPUFaceImageMapper::GPUFaceImageMapper(PixelationFaceImage faceBetas,
        PixelationFace face) {

    __configured = false;
    __useCameraOrFaceBetas = false;
    __faceBetas = faceBetas;
    __face = face;
}


GPUFaceImageMapper::~GPUFaceImageMapper() {
    // nothing to do
}


void GPUFaceImageMapper::configure() {

    if(!__inputImgSet) {
        std::cerr << "ERROR: GPUFaceImageMapper::configure(): inputImage has not been set" << std::endl;
        throw std::exception();
    }

    // if a camera object is passed, compute
    // interpolation coordinates and upload to GPU
    if(__useCameraOrFaceBetas) {

        // transfer faceBetas to GPU memory space
        __faceBetas = PixelationFaceImage(__face.height(),
            __face.width(), 2, FLOAT32);

        // computes face interpolation coordinates
        Image<float> faceBetas = castCoordinates(
            __face.sphericalCoordinates(),
            __camera.getHostCamera()->sphericalCoordinates(),
            __camera.getHostCamera()->isVerticalFlipped());    

        __faceBetas.upload(faceBetas);
    }

    // allocates output image
    __outputImg = PixelationFaceImage(__face.height(), __face.width(),
        __inputImg.depth(), __inputImg.pixelType());


    // grid and block
    __block = dim3(32, 32, 1);
    configureKernelGrid(__face.height(), __face.width(), __block, __grid);

    __configured = true;
}


void GPUFaceImageMapper::compute() {

    startTiming();

    if(!__configured) {
        std::cerr << "ERROR: GPUFaceImageMapper::compute() stage not configured." << std::endl;
        exit(-1);
    }

    // interpolate image
    switch(__inputImg.pixelType()) {

        case UINT8:

            switch(__inputImg.depth()) {
                case 1:
                    // std::cout << "GPUFaceImageMapper::compute(): UINT8 depth 1" << std::endl;
                    interpolateImage_k<unsigned char><<<__grid, __block, 0, __stream>>>(
                        __inputImg.wrap<unsigned char>(),
                        __faceBetas.wrap<float2>(),
                        __outputImg.wrap<unsigned char>(),
                        0);
                    break;

                case 4:
                    // std::cout << "GPUFaceImageMapper::compute(): UINT8 depth 4" << std::endl;
                    interpolateImage_k<uchar4><<<__grid, __block, 0, __stream>>>(
                            __inputImg.wrap<uchar4>(),
                            __faceBetas.wrap<float2>(),
                            __outputImg.wrap<uchar4>(),
                            make_uchar4(0, 0, 0, 0));
                    break;

                default:
                    std::cerr << "ERROR: GPUFaceImageMapper::compute(): unsupported UINT8 channel depth: " << __inputImg.depth() << std::endl;
                    throw std::exception();
            }

            break;

        case FLOAT32:

            switch(__inputImg.depth()) {

                case 1:
                    // std::cout << "GPUFaceImageMapper::compute(): FLOAT32 depth 1" << std::endl;
                    interpolateImage_k<float><<<__grid, __block, 0, __stream>>>(
                        __inputImg.wrap<float>(),
                        __faceBetas.wrap<float2>(),
                        __outputImg.wrap<float>(),
                        0.0f);
                    break;

                case 4:
                    // std::cout << "GPUFaceImageMapper::compute(): FLOAT32 depth 4" << std::endl;
                    interpolateImage_k<float4><<<__grid, __block, 0, __stream>>>(
                        __inputImg.wrap<float4>(),
                        __faceBetas.wrap<float2>(),
                        __outputImg.wrap<float4>(),
                        make_float4(0, 0, 0, 0));
                    break;

                default:
                    std::cerr << "ERROR: GPUFaceImageMapper::compute(): unsupported UINT8 channel depth: " << __inputImg.depth() << std::endl;
                    throw std::exception();
            }

            break;

        default:
            std::cerr << "ERROR: GPUFaceImageMapper::compute(): unsupported pixel type: " << __inputImg.pixelType() << std::endl;
            throw std::exception();

    }

    stopTiming();
}


void GPUFaceImageMapper::setInputImage(PixelationFaceImage inputImg) {

    // configure input image with linear interpolation
    __inputImg = PixelationFaceImage(inputImg, cudaFilterModeLinear, false);
    // __inputImg = inputImg;
    __inputImgSet = true;
}


PixelationFaceImage GPUFaceImageMapper::getMappedImage() {
    return __outputImg;
}

PixelationFaceImage GPUFaceImageMapper::getInterpolationCoordinates() {
    return __faceBetas;
}

} // namespace gpu
} // namespace spherepix