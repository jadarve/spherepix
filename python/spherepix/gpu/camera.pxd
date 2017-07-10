"""
    spherepix.gpu.camera
    --------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

from libcpp.memory cimport shared_ptr

cimport spherepix.camera as scam

cimport spherepix.gpu.pixelation as gpix

cdef extern from 'spherepix/gpu/camera.h' namespace 'spherepix::gpu':
    
    cdef cppclass GPUCamera_cpp 'spherepix::gpu::GPUCamera':

        GPUCamera_cpp()
        GPUCamera_cpp(shared_ptr[scam.Camera_cpp] hostCamera)

        void configure()

        shared_ptr[scam.Camera_cpp] getHostCamera()

        int height() const
        int width() const



    cdef cppclass GPUFaceImageMapper_cpp 'spherepix::gpu::GPUFaceImageMapper':

        GPUFaceImageMapper_cpp()
        GPUFaceImageMapper_cpp(GPUCamera_cpp camera, gpix.PixelationFace_cpp face)
        GPUFaceImageMapper_cpp(gpix.PixelationFaceImage_cpp faceBetas, gpix.PixelationFace_cpp face);

        void configure()
        void compute()
        float elapsedTime()

        # Pipeline stage inputs
        void setInputImage(gpix.PixelationFaceImage_cpp inputImg)

        # Pipeline stage outputs
        gpix.PixelationFaceImage_cpp getMappedImage()
        gpix.PixelationFaceImage_cpp getInterpolationCoordinates()


cdef class GPUCamera:
    
    cdef GPUCamera_cpp cam


cdef class GPUFaceImageMapper:
    
    cdef GPUFaceImageMapper_cpp mapper