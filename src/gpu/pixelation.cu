/**
 * \file pixelation.cu
 * \brief Classes and structures to access pixelation data on GPU functions.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>

#include "spherepix/geometry.h"
#include "spherepix/pixelation.h"

#include "spherepix/gpu/image.h"
#include "spherepix/gpu/pixelation.h"
#include "spherepix/gpu/util.h"

#include "spherepix/gpu/device/util_k.h"

namespace spherepix {
namespace gpu {

using namespace flowfilter::gpu;

std::size_t pixelSize(PixelType t) {
    switch(t) {
        case UINT8:
            return sizeof(char);
        case FLOAT32:
            return sizeof(float);
        default:
            std::cerr << "ERROR: pixelSize(): reached end of switch" << std::endl;
            return 0;
    }
}

cudaChannelFormatKind pixelFormat(PixelType t) {

    switch(t) {
        case UINT8:
            return cudaChannelFormatKindUnsigned;
        case FLOAT32:
            return cudaChannelFormatKindFloat;
        default:
            std::cerr << "ERROR: pixelFormat(): reached end of switch" << std::endl;
            return cudaChannelFormatKindFloat;       
    }
}


//###############################################
// PixelationFace
//###############################################

PixelationFace::PixelationFace() {
    __configured = false;
}


PixelationFace::PixelationFace(Image<float>& etas) {

    __configured = false;
    configure(etas);
}


PixelationFace::~PixelationFace() {

}


void PixelationFace::configure(Image<float>& etas) {

    __hostEtas = etas;

    const int height = etas.height();
    const int width = etas.width();

    //#################################
    // Spherical coordinates
    //#################################

    // device etas
    __etas = PixelationFaceImage(height, width, 4, FLOAT32);
    image3ToGPUImage4(etas, __etas.wrapImage<float4>());

    //#################################
    // Bpix coordinates
    //#################################
    Image<float> Bpix_row0(height, width, 3);
    Image<float> Bpix_row1(height, width, 3);

    // computes Bpix matrix on CPU space and stores each row
    // in a separate image
    betapixMatrixField(etas, Bpix_row0, Bpix_row1);

    // transforms Bpix_row0 from float3 to float4
    __Bpix_row0 = PixelationFaceImage(height, width, 4, FLOAT32);
    __Bpix_row1 = PixelationFaceImage(height, width, 4, FLOAT32);

    image3ToGPUImage4(Bpix_row0, __Bpix_row0.wrapImage<float4>());
    image3ToGPUImage4(Bpix_row1, __Bpix_row1.wrapImage<float4>());


    // creates __faceWrapped
    __faceWrapped.height = height;
    __faceWrapped.width = width;
    __faceWrapped.etas = __etas.wrap<float4>();
    __faceWrapped.Bpix.row0 = __Bpix_row0.wrap<float4>();
    __faceWrapped.Bpix.row1 = __Bpix_row1.wrap<float4>();

    __configured = true;
}


pixface_t PixelationFace::wrap() {
    return __faceWrapped;
}


Image<float> PixelationFace::sphericalCoordinates() {
    return __hostEtas;
}

Image<float> PixelationFace::downloadSphericalCoordinates() {

    Image<float> scoords(height(), width(), __etas.depth());
    __etas.download(scoords);

    return scoords;
}

int PixelationFace::height() const {
    return __etas.height();
}


int PixelationFace::width() const {
    return __etas.width();
}


float PixelationFace::pixelSeparation() const {
    return spherepix::pixelSeparation(__hostEtas);
}



//###############################################
// PixelationFaceImage
//###############################################

PixelationFaceImage::PixelationFaceImage() {

}


PixelationFaceImage::PixelationFaceImage(const int height,
    const int width, const int depth, PixelType pixtype) :
    
    __pixelType(pixtype),
    __img(height, width, depth, pixelSize(pixtype)),
    __imgTexture(__img, pixelFormat(__pixelType)) {
}

PixelationFaceImage::PixelationFaceImage(PixelationFaceImage img,
        cudaTextureFilterMode filterMode,
        const bool normalizedCoordinates) :

    __img(img.__img),
    __pixelType(img.__pixelType),
    __imgTexture(__img, pixelFormat(__pixelType), cudaAddressModeClamp, filterMode, cudaReadModeElementType, normalizedCoordinates) {

}

PixelationFaceImage::~PixelationFaceImage() {
    // nothing to do
}

void PixelationFaceImage::upload(spherepix::image_t& img) {

    flowfilter::image_t imgDesc;
    imgDesc.height = img.height;
    imgDesc.width = img.width;
    imgDesc.depth = img.depth;
    imgDesc.pitch = img.pitch;
    imgDesc.itemSize = img.itemSize;
    imgDesc.data = img.ptr;

    __img.upload(imgDesc);
}

void PixelationFaceImage::download(spherepix::image_t& img) {

    flowfilter::image_t imgDesc;
    imgDesc.height = img.height;
    imgDesc.width = img.width;
    imgDesc.depth = img.depth;
    imgDesc.pitch = img.pitch;
    imgDesc.itemSize = img.itemSize;
    imgDesc.data = img.ptr;

    __img.download(imgDesc);
}

void PixelationFaceImage::copyFrom(PixelationFaceImage& img) {
    __img.copyFrom(img.__img);
}

void PixelationFaceImage::clear() {
    __img.clear();
}

int PixelationFaceImage::height() const {
    return __img.height();
}


int PixelationFaceImage::width() const {
    return __img.width();
}


int PixelationFaceImage::depth() const {
    return __img.depth();
}


int PixelationFaceImage::pitch() const {
    return __img.pitch();
}


int PixelationFaceImage::itemSize() const {
    return __img.itemSize();
}

PixelType PixelationFaceImage::pixelType() const {
    return __pixelType;
}


std::vector<PixelationFace> createFacePyramid(PixelationFace& face, const int levels) {

    std::vector<PixelationFace> facePyr(levels);

    Image<float> etasHost = face.sphericalCoordinates();

    // level zero is a copy of face
    facePyr[0] = PixelationFace{etasHost};

    for(int h = 1; h < levels; h ++) {

        PixelationFace f = facePyr[h-1];

        //  get hsot spherical coordinates (float3)
        Image<float> etas = f.sphericalCoordinates();

        Image<float> etasLevel{etas.height()/2, etas.width()/2, 3};

        for(int r = 0; r < etasLevel.height(); r ++) {
            for(int c = 0; c < etasLevel.width(); c ++) {

                // subsampled eta from previous level
                Eigen::Vector3f eta = readVector3f(etas, 2*r, 2*c);

                // write eta to this level
                writeVector3f<true>(eta, r, c, etasLevel);
            }
        }

        // create a new PixelationFace with etasLevel
        PixelationFace newFace {etasLevel};
        facePyr[h] = newFace;
    }

    return facePyr;
}


} // namespace gpu
} // namespace spherepix