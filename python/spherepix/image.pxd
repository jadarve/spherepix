
cdef extern from 'spherepix/image.h' namespace 'spherepix':


    ctypedef struct image_t_cpp 'spherepix::image_t':

        size_t height
        size_t width
        size_t depth
        size_t pitch
        size_t itemSize
        void* ptr

    cdef cppclass Image_cpp 'spherepix::Image'[T]:

        Image_cpp()
        Image_cpp(const int height, const int width, const int depth)
        Image_cpp(const int height, const int width, const int depth, T* data)

        int height() const
        int width() const
        int depth() const 
        int pitch() const
        int length() const
        int sizeBytes() const

        T* data()

        void copyFrom(const Image_cpp[T]& img)

        T& operator[](size_t idx)


    void interpolateImage_cpp 'spherepix::interpolateImage'[T](const Image_cpp[T]& inputImage,
                                                               const Image_cpp[float]& coordinates,
                                                               Image_cpp[T]& outputImage);

    #######################################################
    # CONVOLUTION METHODS
    #######################################################
    
    Image_cpp[T] convolve2D_cpp 'spherepix::convolve2D'[T](const Image_cpp[T]& img, const Image_cpp[T]& mask)
    void convolve2D_cpp 'spherepix::convolve2D'[T](const Image_cpp[T]& img, const Image_cpp[T]& mask, Image_cpp[T]& output)

    Image_cpp[T] convolveRow_cpp 'spherepix::convolveRow'[T](const Image_cpp[T]& img, const Image_cpp[T]& mask)
    void convolveRow_cpp 'spherepix::convolveRow'[T](const Image_cpp[T]& img, const Image_cpp[T]& mask, Image_cpp[T]& output)

    Image_cpp[T] convolveColumn_cpp 'spherepix::convolveColumn'[T](const Image_cpp[T]& img, const Image_cpp[T]& mask)
    void convolveColumn_cpp 'spherepix::convolveColumn'[T](const Image_cpp[T]& img, const Image_cpp[T]& mask, Image_cpp[T]& output)


cdef class Image:

    cdef object numpyArray
    cdef image_t_cpp img;


cdef class Image_float32:
    
    # holds an np.ndarray object in case it is passed to
    # the constructor of the class
    cdef object numpyArray

    cdef Image_cpp[float] img


cdef class Image_int32:
    
    # holds an np.ndarray object in case it is passed to
    # the constructor of the class
    cdef object numpyArray

    cdef Image_cpp[int] img
    