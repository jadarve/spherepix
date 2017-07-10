"""
    spherepix.gpu.pixelation
    ------------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

from libcpp.vector cimport vector

cimport spherepix.image as simg

cdef extern from 'spherepix/gpu/pixelation.h' namespace 'spherepix::gpu':
    
    cdef enum PixelType:
        UINT8,
        FLOAT32
    
    cdef cppclass PixelationFaceImage_cpp 'spherepix::gpu::PixelationFaceImage':

        PixelationFaceImage_cpp()
        PixelationFaceImage_cpp(const int height, const int width,
            const int depth, PixelType pixtype)

        void upload(simg.image_t_cpp& img)
        void download(simg.image_t_cpp& img)
        void copyFrom(PixelationFaceImage_cpp& img)
        void clear()

        int height() const
        int width() const
        int depth() const
        int pitch() const
        int itemSize() const


    cdef cppclass PixelationFace_cpp 'spherepix::gpu::PixelationFace':

        PixelationFace_cpp()
        PixelationFace_cpp(simg.Image_cpp[float]& etas)

        void configure(simg.Image_cpp[float]& etas)

        simg.Image_cpp[float] sphericalCoordinates()
        simg.Image_cpp[float] downloadSphericalCoordinates()

        int height() const
        int width() const

        float pixelSeparation() const


    cdef vector[PixelationFace_cpp] createFacePyramid_cpp 'spherepix::gpu::createFacePyramid'(
        PixelationFace_cpp& face, const int levels);


cdef class PixelationFaceImage:
    cdef PixelationFaceImage_cpp img

cdef class PixelationFace:
    cdef PixelationFace_cpp face
