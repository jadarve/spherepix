
from libcpp cimport bool
from libcpp.memory cimport shared_ptr

cimport image
cimport eigen

cdef extern from "spherepix/camera.h" namespace "spherepix":

    cdef cppclass Camera_cpp 'spherepix::Camera':
        
        int height() const
        int width() const
        bool isVerticalFlipped() const

        const image.Image_cpp[float] sphericalCoordinates()
        const image.Image_cpp[float] surfaceCoordinates()
        

    cdef cppclass PinholeCamera_cpp 'spherepix::PinholeCamera'(Camera_cpp):

        PinholeCamera_cpp(const float focalLength, const int height, const int width,
            const float sensorHeight, const float sensorWidth)

        PinholeCamera_cpp(const int height, const int width,
            eigen.Matrix3f_cpp& intrinsics)

        int height() const;
        int width() const;
        bool isVerticalFlipped() const

        const image.Image_cpp[float] sphericalCoordinates()
        const image.Image_cpp[float] surfaceCoordinates()

        eigen.Matrix3f_cpp intrinsicsMatrix() const



    cdef cppclass OmnidirectionalCamera_cpp 'spherepix::OmnidirectionalCamera'(Camera_cpp):

        OmnidirectionalCamera_cpp(const image.Image_cpp[float]& sphereCoordinates,
            const bool isVerticalFlipped)

        int height() const;
        int width() const;

        const image.Image_cpp[float] sphericalCoordinates()
        const image.Image_cpp[float] surfaceCoordinates()
        bool isVerticalFlipped() const


cdef class Camera:
    cdef shared_ptr[Camera_cpp] cam

cdef class PinholeCamera(Camera):
    pass

cdef class OmnidirectionalCamera(Camera):
    pass