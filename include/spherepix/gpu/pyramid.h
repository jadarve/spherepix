/**
 * \file pyramid.h
 * \brief Classes for computing image pyramid.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */


#ifndef SPHEREPIX_GPU_PYRAMID_H_
#define SPHEREPIX_GPU_PYRAMID_H_

#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include <flowfilter/gpu/pipeline.h>

#include "spherepix/gpu/pixelation.h"


namespace spherepix {
namespace gpu {


class ImagePyramid : public flowfilter::gpu::Stage {


public:
    ImagePyramid();
    ImagePyramid(spherepix::gpu::PixelationFace face);
    ImagePyramid(spherepix::gpu::PixelationFace face,
        spherepix::gpu::PixelationFaceImage inputImage,
        const int levels);

    ~ImagePyramid();

public:
    /**
     * \brief configures the stage.
     *
     * After configuration, calls to compute()
     * are valid.
     * Input buffers should not change after
     * this method has been called.
     */
    void configure();

    /**
     * \brief perform computation
     */
    void compute();


    //#########################
    // Stage inputs
    //#########################
    void setInputImage(spherepix::gpu::PixelationFaceImage inputImage);


    //#########################
    // Stage outputs
    //#########################
    spherepix::gpu::PixelationFaceImage getImage(const int level);


    //#########################
    // Parameters
    //#########################
    void setLevels(const int levels);
    int getLevels() const;


private:
    bool __configured;
    bool __inputImageSet;

    int __levels;

    spherepix::gpu::PixelationFace __face;

    spherepix::gpu::PixelationFaceImage __inputImage;

    // downsampled images in beta-1 and beta-2 directions
    std::vector<spherepix::gpu::PixelationFaceImage> __pyramidB1;
    std::vector<spherepix::gpu::PixelationFaceImage> __pyramidB2;

    std::vector<dim3> __gridB1;
    std::vector<dim3> __gridB2;

    dim3 __block;
};

} // namespace gpu
} // namespace spherepix

#endif /* SPHEREPIX_GPU_PYRAMID_H_ */