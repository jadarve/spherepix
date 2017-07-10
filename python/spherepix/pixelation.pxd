from libcpp cimport bool

from libcpp.vector cimport vector
from libcpp.pair cimport pair

cimport image

cdef extern from 'spherepix/pixelation.h' namespace 'spherepix':

    cdef enum PixelationMode:
        MODE_1 = 1, # 1 checker face per cube face.
        MODE_2 = 2, # 2 horizontal checker faces per cube face.
        MODE_3 = 3  # 4 checker faces per cube face.


    cdef enum FaceNeighbor:
        LEFT = 0,
        TOP = 1,
        RIGHT = 2,
        BOTTOM = 3


    float pixelSeparation_cpp 'spherepix::pixelSeparation'(const image.Image_cpp[float]& etas)
    
    image.Image_cpp[float] createFace_equidistant_cpp 'spherepix::createFace_equidistant'(
        const int N)

    image.Image_cpp[float] createFace_equiangular_cpp 'spherepix::createFace_equiangular'(
        const int N)

    image.Image_cpp[float] regularizeCoordinates_cpp 'spherepix::regularizeCoordinates'(
        const image.Image_cpp[float]& etas,
        const float Ls,
        const int I)

    vector[image.Image_cpp[float]] createFace_cpp ' spherepix::createFace'(
        PixelationMode mode,
        const int N,
        const int I,
        const float Ls,
        const float M3_theta)

    vector[image.Image_cpp[float]] createCubeFaces_cpp 'spherepix::createCubeFaces'(
        vector[image.Image_cpp[float]]& face)

    const image.Image_cpp[int] getFaceConnectivityGraph_cpp 'spherepix::getFaceConnectivityGraph'(
        PixelationMode mode)

    vector[pair[image.Image_cpp[float], image.Image_cpp[float]]] faceInterpolationBelts_cpp 'spherepix::faceInterpolationBelts'(
        vector[image.Image_cpp[float]]& faceList,
        const int faceIndex,
        const int beltWidth,
        PixelationMode mode)

    pair[image.Image_cpp[float], image.Image_cpp[float]] interpolationBelt_0_cpp 'spherepix::interpolationBelt_0'(
        vector[image.Image_cpp[float]]& faceList,
        const int faceIndex,
        const int beltWidth,
        PixelationMode mode)

    pair[image.Image_cpp[float], image.Image_cpp[float]] interpolationBelt_1_cpp 'spherepix::interpolationBelt_1'(
        vector[image.Image_cpp[float]]& faceList,
        const int faceIndex,
        const int beltWidth,
        PixelationMode mode)

    pair[image.Image_cpp[float], image.Image_cpp[float]] interpolationBelt_2_cpp 'spherepix::interpolationBelt_2'(
        vector[image.Image_cpp[float]]& faceList,
        const int faceIndex,
        const int beltWidth,
        PixelationMode mode)

    pair[image.Image_cpp[float], image.Image_cpp[float]] interpolationBelt_3_cpp 'spherepix::interpolationBelt_3'(
        vector[image.Image_cpp[float]]& faceList,
        const int faceIndex,
        const int beltWidth,
        PixelationMode mode)


    cdef cppclass Pixelation_cpp 'spherepix::Pixelation':

        Pixelation_cpp()

        PixelationMode mode() const

        int faceHeight() const
        int faceWidth() const
        int interpolationBeltWidth() const
        int faceCount() const

        const image.Image_cpp[int] faceConnectivityGraph() const

        const image.Image_cpp[float] faceCoordinates(const int faceIndex)

        const image.Image_cpp[float] interpolationCoordinates(const int faceIndex, FaceNeighbor neighbor) const

        const image.Image_cpp[float] sphericalInterpolationCoordinates(const int faceIndex, FaceNeighbor neighbor) const


    cdef cppclass SphericalImage_cpp 'spherepix::SphericalImage'[T]:
        SphericalImage_cpp()
        SphericalImage_cpp(const Pixelation_cpp& pix, const int depth)

        int faceHeight() const
        int faceWidth() const
        int depth() const
        int faceCount() const

        void clear()

        image.Image_cpp[T]& operator[](size_t idx)

    #######################################################
    # Wrapper for static methods of PixelationFactory class
    #######################################################
    Pixelation_cpp createPixelation_cpp 'spherepix::PixelationFactory::createPixelation'(
        PixelationMode mode,
        const int N,
        const int interpolationBeltWidth,
        const int springIterations,
        const float extraSeparation,
        const float M3_theta,
        const float dt,
        const float M,
        const float C,
        const float K)

    Pixelation_cpp createPixelation_cpp 'spherepix::PixelationFactory::createPixelation'(
        PixelationMode mode,
        vector[image.Image_cpp[float]]& faceList,
        const int interpolationBeltWidth)

    
    #######################################################
    # CONVOLUTION METHODS
    #######################################################

    SphericalImage_cpp[T] convolve2D_cpp 'spherepix::convolve2D'[T](const Pixelation_cpp& pix,
        const SphericalImage_cpp[T]& img,
        const image.Image_cpp[T]& mask)

    SphericalImage_cpp[T] convolveRow_cpp 'spherepix::convolveRow'[T](const Pixelation_cpp& pix,
        const SphericalImage_cpp[T]& img,
        const image.Image_cpp[T]& mask)


    SphericalImage_cpp[T] convolveColumn_cpp 'spherepix::convolveColumn'[T](const Pixelation_cpp& pix,
        const SphericalImage_cpp[T]& img,
        const image.Image_cpp[T]& mask)


    #void convolveFaceRow[T](const Pixelation_cpp& pix, const SphericalImage_cpp[T]& sphericalImg,
    #                        const int faceIndex, const image.Image_cpp[T]& mask, image.Image_cpp[T]& output);


    #void convolveFaceColumn[T](const Pixelation_cpp& pix, const SphericalImage_cpp[T]& sphericalImg,
    #                           const int faceIndex, const image.Image_cpp[T]& mask, image.Image_cpp[T]& output);

    
    #######################################################
    # COORDINATE CASTING
    #######################################################

    image.Image_cpp[float] castCoordinates_cpp 'spherepix::castCoordinates'(
        const image.Image_cpp[float]& etas_0,
        const image.Image_cpp[float]& etas_1,
        const bool flipVertical)



cdef class Pixelation:
    cdef Pixelation_cpp pix


cdef class SphericalImage_float32:
    cdef SphericalImage_cpp[float] img


cdef class SphericalImage_int32:
    cdef SphericalImage_cpp[int] img
