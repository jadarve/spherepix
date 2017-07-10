/**
 * \file camera.h
 * \brief GPU camera classes.
 * \copyright 2015, Juan David Adarve, ANU. See AUTHORS for more details
 * \license 3-clause BSD, see LICENSE for more details
 */

#ifndef SPHEREPIX_GPU_CAMERA_H_
#define SPHEREPIX_GPU_CAMERA_H_

#include <memory>

#include <cuda.h>
#include <cuda_runtime.h>

#include <flowfilter/gpu/image.h>
#include <flowfilter/gpu/pipeline.h>

#include "spherepix/image.h"
#include "spherepix/camera.h"

#include "spherepix/gpu/pixelation.h"

namespace spherepix {
namespace gpu {

class GPUCamera {

public:
    GPUCamera();
    GPUCamera(std::shared_ptr<spherepix::Camera> hostCamera);
    ~GPUCamera();

public:
    void configure();

    std::shared_ptr<spherepix::Camera> getHostCamera();

    int height() const;
    int width() const;


private:

    std::shared_ptr<spherepix::Camera> __hostCamera;

    /** spherical coordinates for each pixel */
    spherepix::gpu::PixelationFaceImage __etas;

};


/**
 * \brief Class to map images taken with a camera to spherical face images
 */
class GPUFaceImageMapper : public flowfilter::gpu::Stage {

public:
    GPUFaceImageMapper();
    
    GPUFaceImageMapper(spherepix::gpu::GPUCamera camera,
        spherepix::gpu::PixelationFace face);

    GPUFaceImageMapper(spherepix::gpu::PixelationFaceImage faceBetas,
        spherepix::gpu::PixelationFace face);

    ~GPUFaceImageMapper();

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
    void setInputImage(spherepix::gpu::PixelationFaceImage inputImg);

    //#########################
    // Stage outputs
    //#########################
    spherepix::gpu::PixelationFaceImage getMappedImage();

    spherepix::gpu::PixelationFaceImage getInterpolationCoordinates();


private:
    // tells wether a camera or a facebetas buffer
    // is passed to the constructor
    bool __useCameraOrFaceBetas;
    bool __configured;
    bool __inputImgSet;

    /** camera used to take image */
    spherepix::gpu::GPUCamera __camera;

    /** pixelation face on which the image is mapped */
    spherepix::gpu::PixelationFace __face;

    /** beta interpolation coordinates */
    spherepix::gpu::PixelationFaceImage __faceBetas;

    /** input image */
    spherepix::gpu::PixelationFaceImage __inputImg;

    /** output image */
    spherepix::gpu::PixelationFaceImage __outputImg;

    // block and grid size for kernel calls
    dim3 __block;
    dim3 __grid;

};

} // namespace gpu
} // namespace spherepix

#endif // SPHEREPIX_GPU_CAMERA_H_