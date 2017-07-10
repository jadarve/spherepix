from libcpp.memory cimport shared_ptr

import numpy as np
cimport numpy as np

from camera cimport Camera, PinholeCamera, OmnidirectionalCamera

import image
cimport image

import eigen
cimport eigen


cdef class Camera:

    def __init__(self):
        """Abstract camera object.

        See also
        --------
        PinholeCamera : Perspective camera class
        OmnidirectionalCamera : Omnidirectional camera class
        """
        # nothing to do
        pass
    
    def __cinit__(self):
        # nothing to do
        pass

    def __dealloc__(self):
        #nothing to do
        pass


    def sphericalCoordinates(self):
        
        cdef image.Image_float32 coords = image.Image_float32()
        coords.img = self.cam.get().sphericalCoordinates()
        return coords.toNumpy()

    def surfaceCoordinates(self):
        
        cdef image.Image_float32 coords = image.Image_float32()
        coords.img = self.cam.get().surfaceCoordinates()
        return coords.toNumpy()


    property height:
        def __get__(self):
            return self.cam.get().height()

    property width:
        def __get__(self):
            return self.cam.get().width()

    property isVerticalFlipped:
        def __get__(self):
            return self.cam.get().isVerticalFlipped();


cdef class PinholeCamera:

    def __init__(self, focalLength=0, resolution=None, sensorSize=None,
        intrinsics=None):
        """Creates a new PinholeCamera object.

        There are two alternatives to create the camera model:
        by providing focalLength, resolution and sensorSize parameters
        to derive the intrinsics matrix, or by directly giving the 3x3
        intrinsics matrix and resolution in pixels.

        if intrinsics is not None, then the camera is created using the
        provided intrinsics matrix, otherwise, focalLenght, resolution and
        sensorSize parameters are used.

        Parameters
        ----------
        focalLength : float, optional.
            Camera focal length in millimeters. Defaults to 0.

        resolution : 2-tuple, optional.
            Resolution (height, width) in pixels. Defaults to None.

        sensorSize : 2-tuple, optional.
            Sensor size (height, width) in millimiters. Defaults to None.

        intrinsics : ndarray
            3x3 numpy ndarray with the intrinsics matrix. Defaults to None.
        """
        pass

    
    def __cinit__(self, focalLength=0, resolution=None, sensorSize=None,
        intrinsics=None):


        cdef PinholeCamera_cpp* thiscam = NULL
        cdef eigen.Matrix3f K = eigen.Matrix3f(None)

        if intrinsics is None:

            thiscam = new PinholeCamera_cpp(focalLength,
                resolution[0], resolution[1], sensorSize[0], sensorSize[1])

        else:
            K.fromNumpy(intrinsics)
            thiscam = new PinholeCamera_cpp(resolution[0], resolution[1], K.M)

        self.cam = shared_ptr[Camera_cpp](<Camera_cpp*>thiscam)


    property intrinsics:

        def __get__(self):
            cdef eigen.Matrix3f K = eigen.Matrix3f()
            K.M = (<PinholeCamera_cpp*>self.cam.get()).intrinsicsMatrix()
            return K.toNumpy()

        def __set__(self, value):
            raise RuntimeError('Camera intrinsics cannot be changed this way. Create a new PynholeCamera object')

        def __del__(self):
            # nothing to do
            pass

    
cdef class OmnidirectionalCamera:

    def __init__(self, np.ndarray[np.float32_t, ndim=3, mode='c'] sphericalCoordinates,
        isVerticalFlipped=True):
        """Creates a new ommnidirecitonal camera object with spherical coordinates.

        This camera object holds the spherical coordinates of an arbitrary
        camera (fisheye, cadatioptric, panoramic)

        Parameters
        ----------
        sphericalCoordinates : ndarray.
            2D array of spherical coordinates (x, y, z) for each pixel.

        isVerticalFlipped : bool, optional.
            Tells if the vertical axis (row axis) is flipped, that is,
            if the top-left coorner of the image is stored in pixel (0,0).
            Defaults True.
        """
        pass
    

    def __cinit__(self, np.ndarray[np.float32_t, ndim=3, mode='c'] sphericalCoordinates,
        isVerticalFlipped=True):

        cdef image.Image_float32 scoords = image.wrap(sphericalCoordinates)

        cdef OmnidirectionalCamera_cpp* thiscam = new OmnidirectionalCamera_cpp(scoords.img, isVerticalFlipped)
        self.cam = shared_ptr[Camera_cpp](<Camera_cpp*>thiscam)



def createPanoramicCamera(resolution, inclination, longitude):
    """Creates a panoramic camera.

    Parameters
    ----------
    resolution : tuple.
        Image resolution (rows, cols).

    inclination : tuple.
        (min, max) inclination.

    longitude : tuple.
        (min, max) longitude.

    Returns
    -------
    panCam : OmnidirectionalCamera.
        OmnidirectionalCamera object with the panoramic image coordinates.
    """

    # linear space for inclination and longitude
    incRange = np.linspace(inclination[0], inclination[1], num=resolution[0])
    longRange = np.linspace(longitude[0], longitude[1], num=resolution[1])

    # generate angular coordinates for the panoramic image
    phi, theta = np.meshgrid(longRange, incRange)

    # spherical coordinates
    X = np.sin(theta)*np.cos(phi)
    Y = np.sin(theta)*np.sin(phi)
    Z = np.cos(theta)

    # concatenate spherical coordinates
    panEtas = np.concatenate([p[...,np.newaxis] for p in [X, Y, Z]], axis=2).astype(np.float32)

    # returns and OmnidirecitonalCamera object
    return OmnidirectionalCamera(panEtas)

