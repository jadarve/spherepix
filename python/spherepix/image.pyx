

from libc.string cimport memcpy

import cython
from cython.operator cimport dereference as deref

cimport numpy as np
import numpy as np

cimport image
#from image cimport Image_cpp
#from image cimport Image_float32
#from image cimport Image_int32
#from image cimport interpolateImage_cpp
#from image cimport convolve2D_cpp
#from image cimport convolveRow_cpp
#from image cimport convolveColumn_cpp


def wrap(np.ndarray arr):
    
    if arr.ndim != 2 and arr.ndim != 3:
        raise ValueError('Incorrect array dimensions, expecting 2 or 3, got {0}'.format(arr.ndim))

    if arr.ndim == 2:
        shape = (arr.shape[0], arr.shape[1])
    else:
        shape = (arr.shape[0], arr.shape[1], arr.shape[2])

    if arr.dtype == np.float32:
        return Image_float32(shape, arr)
    elif arr.dtype == np.int32:
        return Image_int32(shape, arr)
    else:
        raise TypeError('unexpected numpy ndarray dtyoe: {0}'.format(arr.dtype))


def copy(np.ndarray arr):

    if arr.ndim != 2 and arr.ndim != 3:
        raise ValueError('Incorrect array dimensions, expecting 2 or 3, got {0}'.format(arr.ndim))

    if arr.ndim == 2:
        shape = (arr.shape[0], arr.shape[1])
    else:
        shape = (arr.shape[0], arr.shape[1], arr.shape[2])

    if arr.dtype == np.float32:
        rArr = Image_float32(shape)
        rArr.fromNumpy(arr)
        return rArr
    elif arr.dtype == np.int32:
        rArr = Image_int32(shape)
        rArr.fromNumpy(arr)
        return rArr
    else:
        raise TypeError('unexpected numpy ndarray dtyoe: {0}'.format(arr.dtype))


def interpolateImage(inputImage, coordinates, outputImage = None):
    
    if outputImage == None:
        depth = 1 if inputImage.ndim == 2 else inputImage.shape[2]
        cshape = coordinates.shape
        shape = (cshape[0], cshape[1]) if depth == 1 else (cshape[0], cshape[1], depth)
        outputImage = np.zeros(shape, dtype=inputImage.dtype)

    if inputImage.dtype == np.float32:
        __interpolateImage_float(inputImage, coordinates, outputImage)

    return outputImage


def __interpolateImage_float(np.ndarray inputImage, np.ndarray coordinates,
                             np.ndarray outputImage):
    # wrap images
    cdef image.Image_float32 input_wrapped = wrap(inputImage)
    cdef image.Image_float32 coords_wrapped = wrap(coordinates)
    cdef image.Image_float32 output_wrapped = wrap(outputImage)
    interpolateImage_cpp[float](input_wrapped.img, coords_wrapped.img, output_wrapped.img)


def convolve2D(img, mask, output=None):

    # TODO: need to check for contiguity flag
    
    if output == None:
        output = np.zeros_like(img)

    if img.dtype == np.float32:
        __convolve2D_float(img, mask, output)

    return output


def __convolve2D_float(np.ndarray img, np.ndarray mask, np.ndarray output):
    # wrap images
    cdef image.Image_float32 img_wrapped = wrap(img)
    cdef image.Image_float32 mask_wrapped = wrap(mask)
    cdef image.Image_float32 output_wrapped = wrap(output)

    convolve2D_cpp[float](img_wrapped.img, mask_wrapped.img, output_wrapped.img)
    return output


def convolveRow(img, mask, output=None):
    
    if output == None:
        output = np.zeros_like(img)

    if img.dtype == np.float32:
        __convolveRow_float(img, mask, output)

    return output


def __convolveRow_float(np.ndarray img, np.ndarray mask, np.ndarray output):
    # wrap images
    cdef image.Image_float32 img_wrapped = wrap(img)
    cdef image.Image_float32 mask_wrapped = wrap(mask)
    cdef image.Image_float32 output_wrapped = wrap(output)

    convolveRow_cpp[float](img_wrapped.img, mask_wrapped.img, output_wrapped.img)
    return output


def convolveColumn(img, mask, output=None):
    
    if output == None:
        output = np.zeros_like(img)

    if img.dtype == np.float32:
        __convolveColumn_float(img, mask, output)

    return output


def __convolveColumn_float(np.ndarray img, np.ndarray mask, np.ndarray output):
    # wrap images
    cdef image.Image_float32 img_wrapped = wrap(img)
    cdef image.Image_float32 mask_wrapped = wrap(mask)
    cdef image.Image_float32 output_wrapped = wrap(output)

    convolveColumn_cpp[float](img_wrapped.img, mask_wrapped.img, output_wrapped.img)
    return output


cdef class Image:
    """Image wrapper class"""

    def __cinit__(self, np.ndarray arr = None):

        if arr == None:
            self.numpyArray = None
            return

        if not arr.flags['C_CONTIGUOUS']:
            raise ValueError('arr must be C_CONTIGUOUS')

        # hold a reference to this numpy array inside this object
        self.numpyArray = arr

        # validate shape
        shape = arr.shape
        ndim = arr.ndim
        if ndim != 2 and ndim != 3:
            raise ValueError('Incorrect number of image dimensions. Expecting 2 or 3: {0}'.format(ndim))

        # populate image_t properties
        self.img.height = shape[0]
        self.img.width = shape[1]
        self.img.depth = shape[2] if ndim == 3 else 1
        self.img.pitch = arr.strides[0]              # first stride corresponds to row pitch
        self.img.itemSize = arr.strides[ndim -1]     # last stride corresponds to item size
        self.img.ptr = <void*>arr.data


    def __dealloc__(self):

        # nothing to do, memory is relased by self.numpyArray
        pass


    property width:
        def __get__(self):
            return self.img.width

        def __set__(self, v):
            raise RuntimeError('Image width cannot be set')

        def __del__(self):
            pass    # nothing to do


    property height:
        def __get__(self):
            return self.img.height

        def __set__(self, v):
            raise RuntimeError('Image height cannot be set')

        def __del__(self):
            pass    # nothing to do


    property depth:
        def __get__(self):
            return self.img.depth

        def __set__(self, v):
            raise RuntimeError('Image depth cannot be set')

        def __del__(self):
            pass    # nothing to do


    property itemSize:
        def __get__(self):
            return self.img.itemSize

        def __set__(self, v):
            raise RuntimeError('Image itemSize cannot be set')

        def __del__(self):
            pass    # nothing to do


    property pitch:
        def __get__(self):
            return self.img.pitch

        def __set__(self, v):
            raise RuntimeError('Image pitch cannot be set')

        def __del__(self):
            pass    # nothing to do


cdef class Image_float32:

    def __cinit__(self, shape = None, np.ndarray arr = None):

        # if shape is None, creates a dummy Image object
        if shape == None:
            self.numpyArray = None
            return

        if len(shape) != 2 and len(shape) != 3:
            print(shape)
            raise ValueError('Incorrect number of image dimensions. Expecting 2 or 3: {0}'.format(len(shape)))

        cdef int height = shape[0]
        cdef int width = shape[1]
        cdef int depth = shape[2] if len(shape) == 3 else 1

        if height <= 0 or width <= 0:
            raise ValueError('Incorrect image size: {0}'.format(shape))

        if len(shape) == 3 and depth <= 0:
            raise ValueError('Incorrect image depth: {0}'.format(depth))

        self.numpyArray = arr

        if self.numpyArray == None:
            # creates an Image object with its own memory buffer
            self.img = Image_cpp[float](height, width, depth)
        else:

            if arr.dtype != np.float32:
                raise ValueError('Numpy array must have dtype float32: got {0}'.format(arr.dtype))

            self.img = Image_cpp[float](height, width, depth, <float*>arr.data)


    def __dealloc__(self):

        # nothing to do
        pass

    #######################################################
    # METHODS
    #######################################################

    def height(self):
        return self.img.height()

    def width(self):
        return self.img.width()

    def depth(self):
        return self.img.depth()

    def sizeBytes(self):
        return self.img.sizeBytes()


    def toNumpy(self, np.ndarray out = None):

        if out == None:
            out = np.zeros(self.shape, dtype=np.float32)
        else:
            if out.dtype != np.float32:
                raise ValueError('Output array must be float32, got {0}'.format(out.dtype))

            # TODO: check shape

        memcpy(<void*>out.data, <void*>self.img.data(), self.img.sizeBytes())
        return out


    def fromNumpy(self, np.ndarray arr):

        if arr.dtype != np.float32:
            raise ValueError('Input array must be float32, got {0}'.format(arr.dtype))

        # TODO: check shape

        memcpy(<void*>self.img.data(), <void*>arr.data, self.img.sizeBytes())


    #######################################################
    # PROPERTIES
    #######################################################

    property shape:
        def __get__(self):

            if self.img.depth() == 1:
                return (self.img.height(), self.img.width())
            else:
                return (self.img.height(), self.img.width(), self.img.depth())


        def __set__(self, value):
            raise RuntimeError('Image shape cannot be set')


        def __del__(self):
            # nothing to do
            pass


    #######################################################
    # SPECIAL METHODS
    #######################################################

    def __len__(self):
        return self.img.length()

    def __getitem__(self, key):
        return self.img[key]

    def __setitem__(self, key, value):
        self.img[key] = value



cdef class Image_int32:

    def __cinit__(self, shape = None, np.ndarray arr = None):

        # if shape is None, creates a dummy Image object
        if shape == None:
            self.numpyArray = None
            return

        if len(shape) != 2 and len(shape) != 3:
            print(shape)
            raise ValueError('Incorrect number of image dimensions. Expecting 2 or 3: {0}'.format(len(shape)))

        cdef int height = shape[0]
        cdef int width = shape[1]
        cdef int depth = shape[2] if len(shape) == 3 else 1

        if height <= 0 or width <= 0:
            raise ValueError('Incorrect image size: {0}'.format(shape))

        if len(shape) == 3 and depth <= 0:
            raise ValueError('Incorrect image depth: {0}'.format(depth))

        self.numpyArray = arr

        if self.numpyArray == None:
            # creates an Image object with its own memory buffer
            self.img = Image_cpp[int](height, width, depth)
        else:

            if arr.dtype != np.int32:
                raise ValueError('Numpy array must have dtype int32: got {0}'.format(arr.dtype))

            self.img = Image_cpp[int](height, width, depth, <int*>arr.data)


    def __dealloc__(self):

        # nothing to do
        pass

    #######################################################
    # METHODS
    #######################################################

    def height(self):
        return self.img.height()

    def width(self):
        return self.img.width()

    def depth(self):
        return self.img.depth()

    def sizeBytes(self):
        return self.img.sizeBytes()


    def toNumpy(self, np.ndarray out = None):

        if out == None:
            out = np.zeros(self.shape, dtype=np.int32)
        else:
            if out.dtype != np.int32:
                raise ValueError('Output array must be int32, got {0}'.format(out.dtype))

            # TODO: check shape

        memcpy(<void*>out.data, <void*>self.img.data(), self.img.sizeBytes())
        return out


    def fromNumpy(self, np.ndarray arr):

        if arr.dtype != np.int32:
            raise ValueError('Input array must be int32, got {0}'.format(arr.dtype))

        # TODO: check shape

        memcpy(<void*>self.img.data(), <void*>arr.data, self.img.sizeBytes())


    #######################################################
    # PROPERTIES
    #######################################################

    property shape:
        def __get__(self):

            if self.img.depth() == 1:
                return (self.img.height(), self.img.width())
            else:
                return (self.img.height(), self.img.width(), self.img.depth())


        def __set__(self, value):
            raise RuntimeError('Image shape cannot be set')


        def __del__(self):
            # nothing to do
            pass


    #######################################################
    # SPECIAL METHODS
    #######################################################

    def __len__(self):
        return self.img.length()

    def __getitem__(self, key):
        return self.img[key]

    def __setitem__(self, key, value):
        self.img[key] = value