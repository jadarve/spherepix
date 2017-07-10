"""
    spherepix.gpu.pyramid
    ---------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

cimport spherepix.gpu.pixelation as gpix


cdef extern from 'spherepix/gpu/pyramid.h' namespace 'spherepix::gpu':
    

    cdef cppclass ImagePyramid_cpp 'spherepix::gpu::ImagePyramid':


        ImagePyramid_cpp()
        ImagePyramid_cpp(gpix.PixelationFace_cpp face)
        ImagePyramid_cpp(gpix.PixelationFace_cpp face,
            gpix.PixelationFaceImage_cpp inputImage,
            const int levels)

        void configure()
        void compute()
        float elapsedTime()

        # Stage inputs
        void setInputImage(gpix.PixelationFaceImage_cpp inputImage)

        # Stage outputs
        gpix.PixelationFaceImage_cpp getImage(const int level)

        # Parameters
        void setLevels(const int levels)
        int getLevels() const


cdef class ImagePyramid:
    
    cdef ImagePyramid_cpp pyr
