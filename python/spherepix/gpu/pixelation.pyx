"""
    spherepix.gpu.pixelation
    ------------------------

    :copyright: 2015, Juan David Adarve, ANU. See AUTHORS for more details
    :license: 3-clause BSD, see LICENSE for more details
"""

from libcpp.vector cimport vector


cimport numpy as np
import numpy as np

cimport spherepix.image as simg
import spherepix.image as simg


cdef class PixelationFaceImage:

    def __init__(self, shape=None, PixelType pixtype=FLOAT32):
        """Creates a new GPU face image

        If shape is None, then no GPU memory instantiation is
        performed.

        Parameters
        ---------
        shape : tuple, optional.
            Image shape (height, width), defaults to None.

        pixtype : PixelType, optional
            Pixel type, defaults to FLOAT32
        """
        pass
    
    def __cinit__(self, shape=None, PixelType pixtype=FLOAT32):

        if shape == None:
            return
        
        if len(shape) not in [2, 3]:
            raise ValueError('shape should have 2 or 3 dimentions, got: {0}'.format(len(shape)))

        cdef int height = shape[0]
        cdef int width = shape[1]
        cdef int depth = 1 if len(shape) == 2 else shape[2]

        self.img = PixelationFaceImage_cpp(height, width, depth, pixtype)


    def __dealloc__(self):
        # nothing to do
        pass


    def upload(self, np.ndarray img):
        
        # wrap numpy array in a Image object
        cdef simg.Image img_w = simg.Image(img)

        # transfer image to device memory space
        self.img.upload(img_w.img)


    def download(self, dtype=np.float32, np.ndarray output=None):
        """Download image to numpy array

        Parameters
        ----------
        dtype : numpy dtype
            Numpy dtype of the downloaded image

        output : ndarray, optional
            Output numpy ndarray
        """

        if output == None:
            output = np.zeros(self.shape, dtype=dtype)

        oshape = (output.shape[0], output.shape[1], output.shape[2])

        cdef simg.Image output_w = simg.Image(output)
        self.img.download(output_w.img)
        
        return output


    def copyFrom(self, PixelationFaceImage img):
        self.img.copyFrom(img.img)


    def clear(self):
        self.img.clear()


    property shape:

        def __get__(self):

            if self.img.depth() == 1:
                return (self.img.height(), self.img.width())
            else:
                return (self.img.height(), self.img.width(), self.img.depth())

        def __set__(self, value):
            raise RuntimeError('shape cannot be set')

        def __del__(self):
            pass # nothing to do


    property height:
        def __get__(self):
            return self.img.height()

        def __set__(self, value):
            raise RuntimeError('height cannot be set')

        def __del__(self):
            pass # nothing to do


    property width:
        def __get__(self):
            return self.img.width()

        def __set__(self, value):
            raise RuntimeError('width cannot be set')

        def __del__(self):
            pass # nothing to do


    property depth:
        def __get__(self):
            return self.img.depth()

        def __set__(self, value):
            raise RuntimeError('depth cannot be set')

        def __del__(self):
            pass # nothing to do
    

    property pitch:
        def __get__(self):
            return self.img.pitch()

        def __set__(self, value):
            raise RuntimeError('pitch cannot be set')

        def __del__(self):
            pass # nothing to do


    property itemSize:
        def __get__(self):
            return self.img.itemSize()

        def __set__(self, value):
            raise RuntimeError('itemSize cannot be set')

        def __del__(self):
            pass # nothing to do



cdef class PixelationFace:
    
    def __cinit__(self, np.ndarray[dtype=np.float32_t, ndim=3] etas = None):
        
        # wrap etas into Image_float32
        cdef simg.Image_float32 etas_w = simg.wrap(etas)

        # transfer to face
        self.face = PixelationFace_cpp(etas_w.img)


    def downloadSphericalCoordinates(self):

        cdef simg.Image_float32 etas_w = simg.Image_float32()
        etas_w.img = self.face.downloadSphericalCoordinates()
        return etas_w.toNumpy()


    property shape:
        def __get__(self):
            return (self.height, self.width)

        def __set__(self, value):
            raise RuntimeError('shape cannot be set')

        def __del__(self):
            pass # nothing to do


    property height:
        def __get__(self):
            return self.face.height()

        def __set__(self, value):
            raise RuntimeError('height cannot be set')

        def __del__(self):
            pass # nothing to do


    property width:
        def __get__(self):
            return self.face.width()

        def __set__(self, value):
            raise RuntimeError('width cannot be set')

        def __del__(self):
            pass # nothing to do


    property pixelSeparation:
        def __get__(self):
            return self.face.pixelSeparation()

        def __set__(self, value):
            raise RuntimeError('pixel separation cannot be set')

        def __del__(self):
            pass # nothing to do


def createFacePyramid(PixelationFace face, int levels):
    
    cdef vector[PixelationFace_cpp] facePyr = createFacePyramid_cpp(face.face, levels)

    # iterate over facePyr to create PixelationFace objects
    returnPyr = list()

    for i in range(levels):
        faceTemp = PixelationFace()
        faceTemp.face = facePyr[i]

        returnPyr.append(faceTemp)

    return returnPyr

