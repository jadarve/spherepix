/**
 * \file pyramid.cu
 * \brief Classes for computing image pyramid.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#include <iostream>
#include <exception>

#include <flowfilter/gpu/util.h>

#include "spherepix/gpu/pyramid.h"

#include "spherepix/gpu/device/pyramid_k.h"


namespace spherepix {
namespace gpu {


ImagePyramid::ImagePyramid() {

    __configured = false;
    __inputImageSet = false;
    __levels = 0;
}


ImagePyramid::ImagePyramid(PixelationFace face):
    ImagePyramid() {

    __face = face;
}


ImagePyramid::ImagePyramid(PixelationFace face,
    PixelationFaceImage inputImage,
    const int levels):
    ImagePyramid(face) {


    setInputImage(inputImage);
    setLevels(levels);
    configure();
}


ImagePyramid::~ImagePyramid() {
    // nothing to do
}


void ImagePyramid::configure() {

    if(!__inputImageSet) {
        std::cerr << "ERROR: ImageModel::configure(): input image has not been set" << std::endl;
        throw std::exception();
    }

    int height = __inputImage.height();
    int width = __inputImage.width();
    int depth = __inputImage.depth();
    PixelType pixtype = __inputImage.pixelType();
    
    __block = dim3(32, 32, 1);

    __pyramidB1.resize(__levels -1);
    __gridB1.resize(__levels -1);

    __pyramidB2.resize(__levels);
    __gridB2.resize(__levels);

    __pyramidB2[0] = __inputImage;

    dim3 gb1(0,0,0);
    dim3 gb2(0,0,0);

    configureKernelGrid(height, width, __block, gb2);
    __gridB2[0] = gb2;

    // for levels 0 to H - 2
    for(int h = 0; h < __levels -1; h ++) {

        // downsampling in beta-2 (column)
        width /= 2;
        PixelationFaceImage img_b1(height, width, depth, pixtype);
        img_b1.clear();
        __pyramidB1[h] = img_b1;
        
        configureKernelGrid(height, width, __block, gb1);
        // __gridB1.push_back(gb1);
        __gridB1[h] = gb1;

        // downsampling in beta-1 (row)
        height /= 2;
        PixelationFaceImage img_b2(height, width, depth, pixtype);
        img_b2.clear();
        __pyramidB2[h+1] = img_b2;
        
        configureKernelGrid(height, width, __block, gb2);
        __gridB2[h+1] = gb2;
    }

    __configured = true;
}


void ImagePyramid::compute() {

    startTiming();

    if(!__configured) {
        std::cerr << "ERROR: ImagePyramid::compute() stage not configured." << std::endl;
        exit(-1);
    }

    if(__inputImage.pixelType() == UINT8) {

        for(int h = 0; h < __levels -1; h ++) {

            imageDownB1_k<unsigned char>
                <<<__gridB1[h], __block, 0, __stream>>>(
                    __pyramidB2[h].wrap<unsigned char>(),
                    __pyramidB1[h].wrap<unsigned char>());

            imageDownB2_k<unsigned char>
                <<<__gridB2[h], __block, 0, __stream>>>(
                    __pyramidB1[h].wrap<unsigned char>(),
                    __pyramidB2[h +1].wrap<unsigned char>());
        }
        

    } else if(__inputImage.pixelType() == FLOAT32) {

        for(int h = 0; h < __levels -1; h ++) {

            imageDownB1_k<float>
                <<<__gridB1[h], __block, 0, __stream>>>(
                    __pyramidB2[h].wrap<float>(),
                    __pyramidB1[h].wrap<float>());

            imageDownB2_k<float>
                <<<__gridB2[h], __block, 0, __stream>>>(
                    __pyramidB1[h].wrap<float>(),
                    __pyramidB2[h +1].wrap<float>());
        }
    }

    stopTiming();
}


//#########################
// Stage inputs
//#########################
void ImagePyramid::setInputImage(PixelationFaceImage inputImage) {

    const int height = inputImage.height();
    const int width = inputImage.width();

    if(height != __face.height() || width != __face.width()) {
        std::cerr << "ERROR: ImagePyramid::setInputImage(): input image shape different than pixelation face. "
            << "required: [" << __face.height() << ", " << __face.width() << "] passed: [" 
            << height << ", " << width << "]" << std::endl;
        throw std::exception();
    }

    if(inputImage.depth() != 1) {
        std::cerr << "ERROR: ImagePyramid::setInputImage(): input image should have depth 1, got: " << inputImage.depth() << std::endl;
        throw std::exception();
    }

    __inputImage = inputImage;
    __inputImageSet = true;
}


//#########################
// Stage outputs
//#########################
PixelationFaceImage ImagePyramid::getImage(const int level) {

    if(level < 0 || level >= __levels){
        std::cerr << "ERROR: ImagePyramid::getImage(): level index out of range: " << level << std::endl;
        throw std::exception();
    }

    return __pyramidB2[level];
}


//#########################
// Parameters
//#########################
void ImagePyramid::setLevels(const int levels) {

    if(levels <= 0) {
        std::cerr << "ERROR: ImagePyramid::setLevels(): " <<
            "levels should be greater than zero: " << levels << std::endl;
        throw std::exception();
    }

    __levels = levels;
}


int ImagePyramid::getLevels() const {
    return __levels;
}


} // namespace gpu
} // namespace spherepix
