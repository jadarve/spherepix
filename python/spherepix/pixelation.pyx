
cimport pixelation

import os
import json

from cython.operator cimport dereference as deref, preincrement as inc

from libcpp cimport vector
from libcpp cimport pair

import numpy as np
cimport numpy as np

import image
cimport image

from pixelation cimport PixelationMode
from pixelation cimport FaceNeighbor

from pixelation cimport createFace_equidistant_cpp
from pixelation cimport createFace_equiangular_cpp
from pixelation cimport regularizeCoordinates_cpp
from pixelation cimport createFace_cpp
from pixelation cimport createCubeFaces_cpp
from pixelation cimport getFaceConnectivityGraph_cpp

from pixelation cimport interpolationBelt_0_cpp
from pixelation cimport interpolationBelt_1_cpp
from pixelation cimport interpolationBelt_2_cpp
from pixelation cimport interpolationBelt_3_cpp

from pixelation cimport faceInterpolationBelts_cpp

from pixelation cimport Pixelation
from pixelation cimport createPixelation_cpp

from pixelation cimport SphericalImage_cpp
from pixelation cimport SphericalImage_float32
from pixelation cimport SphericalImage_int32

from pixelation cimport convolve2D_cpp
from pixelation cimport convolveRow_cpp
from pixelation cimport convolveColumn_cpp

from pixelation cimport castCoordinates_cpp
from pixelation cimport pixelSeparation_cpp


def pixelSeparation(np.ndarray[np.float32_t, ndim=3, mode='c'] etas):
    """
    Returns the separation between two pixels in a grid of spherical coordinates

    The separation is computed as the distance between two
    points using orthogonal retraction. The reference points
    are chosen to be at the center of the grid (height/2, width/2)

    Parameters
    ----------
    etas : spherical coordinates grid. (x, y, z) normalized.
        Numpy float32 array.

    Returns
    -------
    separation : pixel separation.
        Float

    """
    
    if etas.shape[2] != 3:
        raise ValueError('input coordinates grid must have depth 3, got {0}'.format(etas.shape[2]))
    
    cdef image.Image_float32 etasWrapped = image.wrap(etas)

    return pixelSeparation_cpp(<const image.Image_cpp[float]&>etasWrapped.img)


def createFace_equidistant(int N = 10):
    """
    Creates an equidistant pixelation of one cube face to the sphere.

    Parameters
    ----------
    N : face side. Integer greater than 1

    Returns
    -------
    face : grid of spherical coordinates (x, y, z) normalized.
        Numpy float32 array of shape [N, N, 3].

    Raises
    ------
    ValueError : if N is less or equal to 1.
    """

    if N <= 1:
        raise ValueError('Face side must be greater than one, got {0}'.format(N))
    
    cdef image.Image_float32 face = image.Image_float32()
    face.img = createFace_equidistant_cpp(N)

    return face.toNumpy()


def createFace_equiangular(int N = 10, PixelationMode mode = int(1)):
    """
    Creates an equidistant pixelation of one cube face to the sphere.

    Parameters
    ----------
    N : face side. Integer greater than 1

    mode : Pixelation mode. Integer in {1, 2, 3}

    Returns
    -------
    face : grid of spherical coordinates (x, y, z) normalized.
        Numpy float32 array of shape [N, N, 3].

    Raises
    ------
    ValueError : if N is less or equal to 1.
    """

    if N <= 1:
        raise ValueError('Face side must be greater than one, got {0}'.format(N))

    cdef image.Image_float32 face = image.Image_float32()
    face.img = createFace_equiangular_cpp(N)

    face_m1 = face.toNumpy()

    if mode == 1:
        return face_m1

    if mode == 2:
        face_m2 = np.copy(face_m1[:N/2, :, :])
        return face_m2

    if mode == 3:
        face_m3 = np.copy(face_m1[:N/2, :N/2, :])
        return face_m3


def regularizeCoordinates(np.ndarray[np.float32_t, ndim=3, mode='c'] etas,
                          float Ls = 4.0, int I = 1000):
    """
    Regularized the input spherical coordinates using the spring dynamics system.

    Parameters
    ----------
    etas : spherical coordinates grid. (x, y, z) normalized.
        Numpy float32 array.

    Ls : Additional separation factor, in pixels, added to the resting
        elongation of the spring.

        The new resting elongation is changed according to

        L = L + (Ls/N)*L

    I : number of iterations to run the spring system. Integer

    Returns
    -------
    etasOut : regularized spherical coordinates. (x, y ,z) normalized.
        Numpy float32 array.

    Raises
    ------
    ValueError : if etas is not a 3-channels 2D array
    """

    if etas.shape[2] != 3:
        raise ValueError('input coordinates grid must have depth 3, got {0}'.format(etas.shape[2]))
    
    cdef image.Image_float32 etasWrapped = image.wrap(etas)
    
    cdef image.Image_float32 etasOut = image.Image_float32()
    etasOut.img = regularizeCoordinates_cpp(etasWrapped.img, Ls, I)
    return etasOut.toNumpy()


def createFace(PixelationMode mode, int N,
               int I = 1000,
               float Ls = 4.0,
               float M3_theta = 0.01*np.pi):
    """
    Creates a face of the pixelation in a given mode

    Parameters
    ----------
    mode : Pixelation mode. Integer in {1, 2, 3}

    N : Face side length in pixels. Integer even.

    I : number of iterations to run the spring system. Default to 1000

    Ls : Additional separation factor, in pixels, added to the resting
        elongation of the spring.

        The new resting elongation is changed according to

        L = L + (Ls/N)*L

    M3_theta : rotation angle, in radians, between oposite checker faces in MODE_3
        of the pixelation. Default to 0.01*PI

    Returns
    -------
    faceList : List of grids with the spehrical coordinates of the face
        pixelation.

    Raises
    ------
    ValueError : if N is not an even itenger number
    """

    if N % 2 != 0:
        raise ValueError('face side must be an even number, got {0}'.format(N))
    
    cdef vector[image.Image_cpp[float]] c_faceList = createFace_cpp(mode, N, I, Ls, M3_theta)

    faceList = list()

    cdef vector[image.Image_cpp[float]].iterator it = c_faceList.begin()
    cdef image.Image_float32 img = image.Image_float32()

    while it != c_faceList.end():
        img.img = deref(it)
        faceList.append(img.toNumpy())
        inc(it)

    return faceList


def createCubeFaces(faceList):
    """
    Creates cube faces given a bace face list (from createFace)

    Parameters
    ----------
    faceList : list of checker faces that makes one face of the cube.

    Raises
    ------
    ValueError : if any element of facelist is not a 3D numpy float32 array with
        depth equal 3.

    TypeError : if faceList is not an iterable object
    """

    try:
        for etas in faceList:
            if type(etas) != np.ndarray:
                raise ValueError('Expecting numpy ndarray inside facelist, got {0}'.format(type(etas)))

            if etas.dtype != np.float32:
                raise ValueError('Coordinate array must be of type float32, got {0}'.format(etas.dtype))

            if etas.ndim != 3 and etas.shape[2] != 2:
                raise ValueError('Coordinate grid must have 3 dimensions and depth 3')

    except TypeError:
        raise TypeError('faceList is not an iterable object')

    # wrap faceList coordinates with PyImageF objects
    cdef image.Image_float32 etasWrapped = image.Image_float32()
    cdef vector[image.Image_cpp[float]] faceList_internal
    for etas in faceList:
        etasWrapped = image.wrap(etas)
        faceList_internal.push_back(etasWrapped.img)

    cdef vector[image.Image_cpp[float]] c_cubeFaces = createCubeFaces_cpp(faceList_internal)

    cubeFaceList = list()

    cdef vector[image.Image_cpp[float]].iterator it = c_cubeFaces.begin()
    cdef image.Image_float32 img = image.Image_float32()

    while it != c_cubeFaces.end():
        img.img = deref(it)
        cubeFaceList.append(img.toNumpy())
        inc(it)

    return cubeFaceList


def getFaceConnectivityGraph(PixelationMode mode):
    """
    Retruns the face connectivity graph given a pixelation mode.

    Faces in the connectivity graph are indexed as follows

              1
    --------------------
         |        |
       0 |  face  |  2
         |        |
    --------------------
              3

    Parameters
    ----------
    mode : integer {1, 2, 3}

    Returns
    -------
    graph : connectivity graph. Each row contains the connections on
        the 4 sides of a face.
        Numpy int32 array.
    """
    
    cdef image.Image_int32 graph = image.Image_int32()
    graph.img = getFaceConnectivityGraph_cpp(mode)

    return graph.toNumpy();


def faceInterpolationBelts(faceList, int faceIndex, int beltWidth, PixelationMode mode):
    """
    Returns the face interpolation belts for a given face in the pixelation.

    Parameters
    ----------
    faceList : list of faces that make the pixelation.
    faceIndex : face index.
    beltWidth : width in pixels of the interpolation belt
    mode : pixelation model.

    Returns
    -------
    betasList : list of arrays with the interpolation coordinates in pixels
        in (row, col) format.
    etasList : list of arrays with the spherical coordinates (x, y, z)
        of the interpolated coordinates.
    """

    # wrap faceList coordinates with PyImageF objects
    cdef image.Image_float32 etasWrapped = image.Image_float32()
    cdef vector[image.Image_cpp[float]] faceList_internal
    for etas in faceList:
        etasWrapped = image.wrap(etas)
        faceList_internal.push_back(etasWrapped.img)

    cdef vector[pair[image.Image_cpp[float], image.Image_cpp[float]]] beltList = \
        faceInterpolationBelts_cpp(faceList_internal, faceIndex, beltWidth, mode)

    
    betasList = list()
    etasList = list()

    cdef image.Image_float32 betasInterp = image.Image_float32()
    cdef image.Image_float32 etasInterp = image.Image_float32()

    cdef vector[pair[image.Image_cpp[float], image.Image_cpp[float]]].iterator it = beltList.begin()
    
    while it != beltList.end():
        
        betasInterp.img = deref(it).first
        etasInterp.img = deref(it).second

        betasList.append(betasInterp.toNumpy())
        etasList.append(etasInterp.toNumpy())

        inc(it)

    return betasList, etasList


def interpolationBelt_0(faceList, int faceIndex, int beltWidth, PixelationMode mode):
    """
    Returns the interpolation belt zero.

    Parameters
    ----------
    faceList : list of spherical coordinates that make the pixelation.
    faceIndex : face index within faceList.
    beltWidth : width in pixels of the interpolation belt.
    mode : pixelation mode.

    Returns
    -------
    betas : pixel interpolation coordinates.
    etas : spherical coordinates of the interpolation belt.
    """
    
    # wrap faceList coordinates with PyImageF objects
    cdef image.Image_float32 etasWrapped = image.Image_float32()
    cdef vector[image.Image_cpp[float]] faceList_internal
    for etas in faceList:
        etasWrapped = image.wrap(etas)
        faceList_internal.push_back(etasWrapped.img)


    cdef pair[image.Image_cpp[float], image.Image_cpp[float]] belt = interpolationBelt_0_cpp(faceList_internal,
        faceIndex, beltWidth, mode)

    cdef image.Image_float32 betasInterp = image.Image_float32()
    cdef image.Image_float32 etasInterp = image.Image_float32()

    betasInterp.img = belt.first;
    etasInterp.img = belt.second;

    return betasInterp.toNumpy(), etasInterp.toNumpy()


def interpolationBelt_1(faceList, int faceIndex, int beltWidth, PixelationMode mode):
    """
    Returns the interpolation belt one.

    Parameters
    ----------
    faceList : list of spherical coordinates that make the pixelation.
    faceIndex : face index within faceList.
    beltWidth : width in pixels of the interpolation belt.
    mode : pixelation mode.

    Returns
    -------
    betas : pixel interpolation coordinates.
    etas : spherical coordinates of the interpolation belt.
    """

    # wrap faceList coordinates with PyImageF objects
    cdef image.Image_float32 etasWrapped = image.Image_float32()
    cdef vector[image.Image_cpp[float]] faceList_internal
    for etas in faceList:
        etasWrapped = image.wrap(etas)
        faceList_internal.push_back(etasWrapped.img)


    cdef pair[image.Image_cpp[float], image.Image_cpp[float]] belt = interpolationBelt_1_cpp(faceList_internal,
        faceIndex, beltWidth, mode)

    cdef image.Image_float32 betasInterp = image.Image_float32()
    cdef image.Image_float32 etasInterp = image.Image_float32()

    betasInterp.img = belt.first;
    etasInterp.img = belt.second;

    return betasInterp.toNumpy(), etasInterp.toNumpy()


def interpolationBelt_2(faceList, int faceIndex, int beltWidth, PixelationMode mode):
    """
    Returns the interpolation belt two.

    Parameters
    ----------
    faceList : list of spherical coordinates that make the pixelation.
    faceIndex : face index within faceList.
    beltWidth : width in pixels of the interpolation belt.
    mode : pixelation mode.

    Returns
    -------
    betas : pixel interpolation coordinates.
    etas : spherical coordinates of the interpolation belt.
    """

    # wrap faceList coordinates with PyImageF objects
    cdef image.Image_float32 etasWrapped = image.Image_float32()
    cdef vector[image.Image_cpp[float]] faceList_internal
    for etas in faceList:
        etasWrapped = image.wrap(etas)
        faceList_internal.push_back(etasWrapped.img)


    cdef pair[image.Image_cpp[float], image.Image_cpp[float]] belt = interpolationBelt_2_cpp(faceList_internal,
        faceIndex, beltWidth, mode)

    cdef image.Image_float32 betasInterp = image.Image_float32()
    cdef image.Image_float32 etasInterp = image.Image_float32()

    betasInterp.img = belt.first;
    etasInterp.img = belt.second;

    return betasInterp.toNumpy(), etasInterp.toNumpy()


def interpolationBelt_3(faceList, int faceIndex, int beltWidth, PixelationMode mode):
    """
    Returns the interpolation belt three.

    Parameters
    ----------
    faceList : list of spherical coordinates that make the pixelation.
    faceIndex : face index within faceList.
    beltWidth : width in pixels of the interpolation belt.
    mode : pixelation mode.

    Returns
    -------
    betas : pixel interpolation coordinates.
    etas : spherical coordinates of the interpolation belt.
    """

    # wrap faceList coordinates with PyImageF objects
    cdef image.Image_float32 etasWrapped = image.Image_float32()
    cdef vector[image.Image_cpp[float]] faceList_internal
    for etas in faceList:
        etasWrapped = image.wrap(etas)
        faceList_internal.push_back(etasWrapped.img)


    cdef pair[image.Image_cpp[float], image.Image_cpp[float]] belt = interpolationBelt_3_cpp(faceList_internal,
        faceIndex, beltWidth, mode)

    cdef image.Image_float32 betasInterp = image.Image_float32()
    cdef image.Image_float32 etasInterp = image.Image_float32()

    betasInterp.img = belt.first;
    etasInterp.img = belt.second;

    return betasInterp.toNumpy(), etasInterp.toNumpy()



def createPixelation(PixelationMode mode, int faceSide,
                     int interpolationBeltWidth,
                     int springIterations = 1000,
                     float extraSeparation = 4.0,
                     float M3_theta = 0.01*np.pi,
                     float dt = 0.2,
                     float M = 1.0,
                     float C = 0.05,
                     float K = 5.0):
    
    cdef Pixelation pix = Pixelation()
    pix.pix = createPixelation_cpp(mode, faceSide,
        interpolationBeltWidth, springIterations, extraSeparation, M3_theta,
        dt, M, C, K)

    return pix

def createPixelationFromFacelist(PixelationMode mode, faceList,
                                 int interpolationBeltWidth):
    
    # wrap faceList coordinates with PyImageF objects
    cdef image.Image_float32 etasWrapped = image.Image_float32()
    cdef vector[image.Image_cpp[float]] faceList_internal
    for etas in faceList:
        etasWrapped = image.wrap(etas)
        faceList_internal.push_back(etasWrapped.img)

    cdef Pixelation pix = Pixelation()
    pix.pix = createPixelation_cpp(mode, faceList_internal, interpolationBeltWidth)

    return pix


def loadPixelation(filename):
    """
    Load pixelation from file.

    Parameters
    ----------
    filename : string
        File path to json file with meta information to read
        pixelation.

    Returns
    -------
    pix : Pixelation
        Sphere pixelation.
    """

    f = open(filename)
    basePath, _ = os.path.split(f.name)

    pixDesc = json.load(f)

    try:
        mode = pixDesc['mode']
        border = pixDesc['border']
        files = pixDesc['files']

        # number of checkerboard faces to read
        # according to pixelation mode
        N = {1 : 1, 2 : 2, 3 : 4}[mode]

        faceList = list()
        for n in range(N):
            faceList.append(
                np.load('{0}/{1}'.format(basePath, files.format(n))))

        return createPixelationFromFacelist(mode, faceList, border)

    except KeyError as e:
        print('ERROR: missing attribute in pixelation file: {0}'.format(e))


def savePixelation(pix, filePath, fileName='pix.json', filePattern='{0:02d}.npy'):
    """
    Save pixelation to folder.

    This method creates a json descriptor for the pixelation properties and
    saves the nd arrays storing face coordinates.

    Parameters
    ----------
    pix : Pixelation.
        Pixelation object to save

    filePath : string.
        Path to folder where pixelation data will be stored.

    fileName : string, optional.
        Name of the pixelation description file. Defaults to 'pix.json'

    filePattern : string, optional.
        String pattern for storing pixelation face coordinates.
        Defaults to '{0:02d}.npy'

    See also
    --------
    loadPixelation() : Load pixelation from file.
    """
    
    pixDesc = dict()
    pixDesc['mode'] = int(pix.mode())
    pixDesc['border'] = pix.interpolationBeltWidth()
    pixDesc['files'] = filePattern
    
    pixFilePath = os.path.join(filePath, fileName)
    
    with open(pixFilePath, 'w') as fp:
        json.dump(pixDesc, fp)
        
    for k in range(pix.faceCount()):
        face = pix.faceCoordinates(k)
        facePath = os.path.join(filePath, filePattern.format(k))
        np.save(facePath, face)


def convolve2D(Pixelation pix, img, mask):
    
    if type(img) == SphericalImage_float32:
        return __convolve2D_float32(pix, img, mask)

def __convolve2D_float32(Pixelation pix, SphericalImage_float32 img, mask):
    cdef image.Image_float32 mask_cpp = image.copy(mask)
    cdef SphericalImage_float32 out = SphericalImage_float32()
    out.img = convolve2D_cpp[float](pix.pix, img.img, mask_cpp.img)
    return out

def convolveRow(Pixelation pix, img, mask):
    
    if type(img) == SphericalImage_float32:
        return __convolveRow_float32(pix, img, mask)


def __convolveRow_float32(Pixelation pix, SphericalImage_float32 img, mask):
    cdef image.Image_float32 mask_cpp = image.copy(mask)
    cdef SphericalImage_float32 out = SphericalImage_float32()
    out.img = convolveRow_cpp[float](pix.pix, img.img, mask_cpp.img)
    return out


def convolveColumn(Pixelation pix, img, mask):
    
    if type(img) == SphericalImage_float32:
        return __convolveColumn_float32(pix, img, mask)


def __convolveColumn_float32(Pixelation pix, SphericalImage_float32 img, mask):
    cdef image.Image_float32 mask_cpp = image.copy(mask)
    cdef SphericalImage_float32 out = SphericalImage_float32()
    out.img = convolveColumn_cpp[float](pix.pix, img.img, mask_cpp.img)
    return out


def castCoordinates(np.ndarray[np.float32_t, ndim=3, mode='c'] etas_0,
                    np.ndarray[np.float32_t, ndim=3, mode='c'] etas_1,
                    flipVertical=True):
    
    # wrap images
    cdef image.Image_float32 etas_0_wrapped = image.wrap(etas_0)
    cdef image.Image_float32 etas_1_wrapped = image.wrap(etas_1)

    cdef image.Image_float32 betas = image.Image_float32()

    betas.img = castCoordinates_cpp(etas_0_wrapped.img, etas_1_wrapped.img, flipVertical)
    return betas.toNumpy()


cdef class Pixelation:

    def __cinit__(self):
         pass


    def mode(self):
        return self.pix.mode()

    def faceHeight(self):
        return self.pix.faceHeight();

    def faceWidth(self):
        return self.pix.faceWidth();

    def interpolationBeltWidth(self):
        return self.pix.interpolationBeltWidth();

    def faceCount(self):
        return self.pix.faceCount();

    def faceConnectivityGraph(self):
        cdef image.Image_int32 connGraph = image.Image_int32()
        connGraph.img = self.pix.faceConnectivityGraph()
        return connGraph.toNumpy()

    def faceCoordinates(self, int faceIndex):
        cdef image.Image_float32 face = image.Image_float32()
        face.img = self.pix.faceCoordinates(faceIndex)
        return face.toNumpy()

    def interpolationCoordinates(self, int faceIndex, FaceNeighbor neighbor):
        cdef image.Image_float32 coords = image.Image_float32()
        coords.img = self.pix.interpolationCoordinates(faceIndex, neighbor)
        return coords.toNumpy()

    def sphericalInterpolationCoordinates(self, int faceIndex, FaceNeighbor neighbor):
        cdef image.Image_float32 coords = image.Image_float32()
        coords.img = self.pix.sphericalInterpolationCoordinates(faceIndex, neighbor)
        return coords.toNumpy()


cdef class SphericalImage_float32:
    
    def __cinit__(self, Pixelation pix=None, int depth=1):
        if pix != None:
            self.img = SphericalImage_cpp[float](pix.pix, depth)

    def faceHeight(self):
        return self.img.faceHeight()

    def faceWidth(self):
        return self.img.faceWidth()

    def depth(self):
        return self.img.depth()

    def numberFaces(self):
        return self.img.faceCount()

    def clear(self):
        self.img.clear()


    #######################################################
    # PROPERTIES
    #######################################################

    property shape:
        def __get__(self):

            if self.img.depth() == 1:
                return (self.img.faceHeight(), self.img.faceWidth())
            else:
                return (self.img.faceHeight(), self.img.faceWidth(), self.img.depth())

        def __set__(self, value):
            raise RuntimeError('SphericalImage_float32 shape cannot be set')

        def __del__(self):
            # nothing to do
            pass

    #######################################################
    # SPECIAL METHODS
    #######################################################
    def __getitem__(self, key):
        cdef image.Image_float32 out = image.Image_float32()
        out.img = self.img[key]
        return out.toNumpy()

    def __setitem__(self, key, value):
        cdef image.Image_float32 imgWrapped = image.wrap(value)
        cdef image.Image_cpp[float]* imgPtr = &self.img[key]
        imgPtr.copyFrom(imgWrapped.img)


cdef class SphericalImage_int32:
    
    def __cinit__(self, Pixelation pix=None, int depth=1):
        if pix != None:
            self.img = SphericalImage_cpp[int](pix.pix, depth)

    def faceHeight(self):
        return self.img.faceHeight()

    def faceWidth(self):
        return self.img.faceWidth()

    def depth(self):
        return self.img.depth()

    def numberFaces(self):
        return self.img.faceCount()

    def clear(self):
        self.img.clear()

    #######################################################
    # PROPERTIES
    #######################################################

    property shape:
        def __get__(self):

            if self.img.depth() == 1:
                return (self.img.faceHeight(), self.img.faceWidth())
            else:
                return (self.img.faceHeight(), self.img.faceWidth(), self.img.depth())


        def __set__(self, value):
            raise RuntimeError('SphericalImage_int32 shape cannot be set')

        def __del__(self):
            # nothing to do
            pass

    #######################################################
    # SPECIAL METHODS
    #######################################################
    def __getitem__(self, key):
        cdef image.Image_int32 out = image.Image_float32()
        out.img = self.img[key]
        return out.toNumpy()

    def __setitem__(self, key, value):
        cdef image.Image_int32 imgWrapped = image.wrap(value)
        cdef image.Image_cpp[int]* imgPtr = &self.img[key]
        imgPtr.copyFrom(imgWrapped.img)