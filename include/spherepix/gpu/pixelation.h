/**
 * \file pixelation.h
 * \brief Classes and structures to access pixelation data on GPU functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#ifndef SPHEREPIX_GPU_PIXELATION_H_
#define SPHEREPIX_GPU_PIXELATION_H_

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

// #include <flowfilter/gpu/pipeline.h>
#include <flowfilter/gpu/image.h>

#include "spherepix/image.h"

namespace spherepix {
namespace gpu {


/**
 * \brief Valid pixel types for PixelationFaceImage
 */
enum PixelType {
    UINT8,
    FLOAT32
};


/**
 * \brief Pixelation face image.
 */
template<typename T>
struct pixfaceimg_t {
    cudaTextureObject_t imgTexture;
    flowfilter::gpu::gpuimage_t<T> img;
};


/**
 * \brief Struct describing a 2D image of 2x3 matrices.
 */
typedef struct {
    pixfaceimg_t<float4> row0;
    pixfaceimg_t<float4> row1;
} pixfaceimgMat23F_t;


/**
 * \brief Struct describing a pixelation face.
 */
typedef struct {
    pixfaceimg_t<float4> etas;
    pixfaceimgMat23F_t Bpix;
    int height;
    int width;
} pixface_t;


/**
 * \brief returns size in bytes of a given pixel type.
 */
std::size_t pixelSize(PixelType t);

/**
 * \brief returns cudaChannelFormatKind for a given pixel type
 */
cudaChannelFormatKind pixelFormat(PixelType t);


class PixelationFaceImage {


public:
    PixelationFaceImage();
    PixelationFaceImage(const int height, const int width,
        const int depth = 1, PixelType pixtype = FLOAT32);

    PixelationFaceImage(PixelationFaceImage img,
        cudaTextureFilterMode filterMode,
        const bool normalizedCoordinates);

    ~PixelationFaceImage();

public:
    template<typename T>
    pixfaceimg_t<T> wrap() {
        
        pixfaceimg_t<T> img;
        img.imgTexture = __imgTexture.getTextureObject();
        img.img = __img.wrap<T>();
        return img;
    }

    template<typename T>
    flowfilter::gpu::gpuimage_t<T> wrapImage() {
        return __img.wrap<T>();
    }

    template<typename T>
    void upload(spherepix::Image<T>& img) {

        flowfilter::image_t imgDesc;
        imgDesc.height = img.height();
        imgDesc.width = img.width();
        imgDesc.depth = img.depth();
        imgDesc.pitch = img.pitch();
        imgDesc.itemSize = sizeof(T);
        imgDesc.data = img.data();

        __img.upload(imgDesc);
    }

    template<typename T>
    void download(spherepix::Image<T>& img) {

        flowfilter::image_t imgDesc;
        imgDesc.height = img.height();
        imgDesc.width = img.width();
        imgDesc.depth = img.depth();
        imgDesc.pitch = img.pitch();
        imgDesc.itemSize = sizeof(T);
        imgDesc.data = img.data();

        __img.download(imgDesc);
    }

    void upload(spherepix::image_t& img);
    void download(spherepix::image_t& img);

    void copyFrom(PixelationFaceImage& img);

    void clear();

    int height() const;
    int width() const;
    int depth() const;
    int pitch() const;
    int itemSize() const;
    PixelType pixelType() const;

private:
    PixelType __pixelType;
    flowfilter::gpu::GPUImage __img;
    flowfilter::gpu::GPUTexture __imgTexture;
};



class PixelationFace {

public:
    PixelationFace();
    PixelationFace(spherepix::Image<float>& etas);
    ~PixelationFace();

public:
    void configure(spherepix::Image<float>& etas);

    spherepix::Image<float> sphericalCoordinates();
    spherepix::Image<float> downloadSphericalCoordinates();

    /**
     * \brief returns the separation between two neighboring pixels
     */
    float pixelSeparation() const;

    int height() const;
    int width() const;

    //###########################################
    // GPU OUTPUTS
    //###########################################
    pixface_t wrap();


private:
    /** host buffer with spherical coordinates */
    spherepix::Image<float> __hostEtas;

    /** check if the face has been configured */
    bool __configured;

    /** wrapped face struct to use in device kernels */
    pixface_t __faceWrapped;

    /** device spherical coordinates */
    PixelationFaceImage __etas;

    /** Mu to beta pix matrix */
    PixelationFaceImage __Bpix_row0;
    PixelationFaceImage __Bpix_row1;
};


std::vector<PixelationFace> createFacePyramid(PixelationFace& face, const int levels);


} // namespace gpu
} // namespace spherepix

#endif // SPHEREPIX_GPU_PIXELATION_H_