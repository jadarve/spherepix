cimport geometry

from geometry cimport projectionMatrix_cpp

from geometry cimport etaToMu_orthographic_cpp
from geometry cimport etaToMu_perspective_cpp
from geometry cimport etaToMu_geodesic_cpp
from geometry cimport etaToMu_cordal_cpp

from geometry cimport muToEta_orthographic_cpp
from geometry cimport muToEta_perspective_cpp
from geometry cimport muToEta_geodesic_cpp
from geometry cimport muToEta_cordal_cpp

from geometry cimport rotationMatrix_cpp
from geometry cimport rotationMatrixAxisAngle_cpp

from geometry cimport getOrthonormalBasis_cpp
from geometry cimport muToBetapix_cpp
from geometry cimport betapixToMu_cpp

from geometry cimport findInterpolationCoodinates_cpp

from geometry cimport retractField_orthographic_cpp
from geometry cimport betapixFieldToMu_cpp, muFieldToBetapix_cpp
from geometry cimport dotProductField3_cpp
from geometry cimport transformCoordinates_cpp
from geometry cimport angleBetweenNeighbors_cpp
from geometry cimport distanceBetweenNeighbors_cpp

import eigen
cimport eigen

import image
cimport image

from libc.string cimport memcpy

import cython
from cython.operator cimport dereference as deref

import numpy as np
cimport numpy as np


###########################################################
# PROJECTION MATRIX
###########################################################

def projectionMatrix(np.ndarray[np.float32_t, ndim=2, mode='c'] eta):
    """
    Returns the tangent space projection matrix for eta.

    Parameters
    ----------
    eta : column vector (x, y, z), normalized.
        Numpy float32 array.

    Returns
    -------
    P : 3x3 projection matrix.
        Numpy float32 array.

    Raises
    ------
    ValueError : if eta is not a 3-column numpy array.
    """

    cdef eigen.Vector3f eta_eigen = eigen.Vector3f(eta)
    cdef eigen.Matrix3f P_eigen = eigen.Matrix3f()
    
    P_eigen.M = projectionMatrix_cpp(eta_eigen.v)
    return P_eigen.toNumpy()


###########################################################
# TANGENT SPACE PROJECTIONS
###########################################################

def etaToMu_orthographic(np.ndarray[np.float32_t, ndim=2, mode='c'] eta_0,
                         np.ndarray[np.float32_t, ndim=2, mode='c'] eta):
    """
    Returns the orthographic retraction of eta on the tangent space of eta_0.

    Parameters
    ----------
    eta_0 : reference point. Column vector (x, y, z), normalized.
        Numpy float32 array

    eta : point to project. Column vector (x, y, z), normalized.
        Numpy float32 array

    Returns
    -------
    mu : projected coordinate. Column vector (x, y, z).
        Numpy float32 array

    Raises
    ------
    ValueError : if either eta_0 or eta are not column vectors.
    """
    
    cdef eigen.Vector3f eta_0_eigen = eigen.Vector3f(eta_0)
    cdef eigen.Vector3f eta_eigen = eigen.Vector3f(eta)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f()

    mu_eigen.v = etaToMu_orthographic_cpp(eta_0_eigen.v, eta_eigen.v)
    return mu_eigen.toNumpy()


def etaToMu_perspective(np.ndarray[np.float32_t, ndim=2, mode='c'] eta_0,
                        np.ndarray[np.float32_t, ndim=2, mode='c'] eta):
    """
    Returns the perspective retraction of eta on the tangent space of eta_0.

    Parameters
    ----------
    eta_0 : reference point. Column vector (x, y, z), normalized.
        Numpy float32 array

    eta : point to project. Column vector (x, y, z), normalized.
        Numpy float32 array

    Returns
    -------
    mu : projected coordinate. Column vector (x, y, z).
        Numpy float32 array

    Raises
    ------
    ValueError : if either eta_0 or eta are not column vectors.
    """
    
    cdef eigen.Vector3f eta_0_eigen = eigen.Vector3f(eta_0)
    cdef eigen.Vector3f eta_eigen = eigen.Vector3f(eta)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f()

    mu_eigen.v = etaToMu_perspective_cpp(eta_0_eigen.v, eta_eigen.v)
    return mu_eigen.toNumpy()


def etaToMu_geodesic(np.ndarray[np.float32_t, ndim=2, mode='c'] eta_0,
                     np.ndarray[np.float32_t, ndim=2, mode='c'] eta):
    """
    Returns the geodesic retraction of eta on the tangent space of eta_0.

    Parameters
    ----------
    eta_0 : reference point. Column vector (x, y, z), normalized.
        Numpy float32 array

    eta : point to project. Column vector (x, y, z), normalized.
        Numpy float32 array

    Returns
    -------
    mu : projected coordinate. Column vector (x, y, z).
        Numpy float32 array

    Raises
    ------
    ValueError : if either eta_0 or eta are not column vectors.
    """
    
    cdef eigen.Vector3f eta_0_eigen = eigen.Vector3f(eta_0)
    cdef eigen.Vector3f eta_eigen = eigen.Vector3f(eta)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f()

    mu_eigen.v = etaToMu_geodesic_cpp(eta_0_eigen.v, eta_eigen.v)
    return mu_eigen.toNumpy()


def etaToMu_cordal(np.ndarray[np.float32_t, ndim=2, mode='c'] eta_0,
                   np.ndarray[np.float32_t, ndim=2, mode='c'] eta):
    """
    Returns the cordal retraction of eta on the tangent space of eta_0.

    Parameters
    ----------
    eta_0 : reference point. Column vector (x, y, z), normalized.
        Numpy float32 array

    eta : point to project. Column vector (x, y, z), normalized.
        Numpy float32 array

    Returns
    -------
    mu : projected coordinate. Column vector (x, y, z).
        Numpy float32 array

    Raises
    ------
    ValueError : if either eta_0 or eta are not column vectors.
    """
    
    cdef eigen.Vector3f eta_0_eigen = eigen.Vector3f(eta_0)
    cdef eigen.Vector3f eta_eigen = eigen.Vector3f(eta)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f()

    mu_eigen.v = etaToMu_cordal_cpp(eta_0_eigen.v, eta_eigen.v)
    return mu_eigen.toNumpy()


###########################################################
# RETRACTIONS
###########################################################

def muToEta_orthographic(np.ndarray[np.float32_t, ndim=2, mode='c'] eta_0,
                         np.ndarray[np.float32_t, ndim=2, mode='c'] mu):
    """
    Returns the orthographic inverse retraction of mu given reference eta_0.
 
    Parameters
    ----------
    eta_0 : reference point. Column vector (x, y, z), normalized.
        Numpy float32 array

    mu : tangent space point. Column vector (x, y, z).
        Numpy float32 array

    Returns
    -------
    eta : coordinate. Column vector (x, y, z), normalized.
        Numpy float32 array

    Raises
    ------
    ValueError : if either eta_0 or mu are not column vectors.
    """

    cdef eigen.Vector3f eta_0_eigen = eigen.Vector3f(eta_0)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f(mu)
    cdef eigen.Vector3f eta_eigen = eigen.Vector3f()

    eta_eigen.v = muToEta_orthographic_cpp(eta_0_eigen.v, mu_eigen.v)
    return eta_eigen.toNumpy()


def muToEta_perspective(np.ndarray[np.float32_t, ndim=2, mode='c'] eta_0,
                        np.ndarray[np.float32_t, ndim=2, mode='c'] mu):
    """
    Returns the perspective inverse retraction of mu given reference eta_0
 
    Parameters
    ----------
    eta_0 : reference point. Column vector (x, y, z), normalized.
        Numpy float32 array

    mu : tangent space point. Column vector (x, y, z).
        Numpy float32 array

    Returns
    -------
    eta : coordinate. Column vector (x, y, z), normalized.
        Numpy float32 array

    Raises
    ------
    ValueError : if either eta_0 or mu are not column vectors.
    """

    cdef eigen.Vector3f eta_0_eigen = eigen.Vector3f(eta_0)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f(mu)
    cdef eigen.Vector3f eta_eigen = eigen.Vector3f()

    eta_eigen.v = muToEta_perspective_cpp(eta_0_eigen.v, mu_eigen.v)
    return eta_eigen.toNumpy()


def muToEta_geodesic(np.ndarray[np.float32_t, ndim=2, mode='c'] eta_0,
                     np.ndarray[np.float32_t, ndim=2, mode='c'] mu):
    """
    Returns the geodesic inverse retraction of mu given reference eta_0.
 
    Parameters
    ----------
    eta_0 : reference point. Column vector (x, y, z), normalized.
        Numpy float32 array

    mu : tangent space point. Column vector (x, y, z).
        Numpy float32 array

    Returns
    -------
    eta : coordinate. Column vector (x, y, z), normalized.
        Numpy float32 array

    Raises
    ------
    ValueError : if either eta_0 or mu are not column vectors.
    """

    cdef eigen.Vector3f eta_0_eigen = eigen.Vector3f(eta_0)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f(mu)
    cdef eigen.Vector3f eta_eigen = eigen.Vector3f()

    eta_eigen.v = muToEta_geodesic_cpp(eta_0_eigen.v, mu_eigen.v)
    return eta_eigen.toNumpy()


def muToEta_cordal(np.ndarray[np.float32_t, ndim=2, mode='c'] eta_0,
                   np.ndarray[np.float32_t, ndim=2, mode='c'] mu):
    """
    Returns the cordal inverse retraction of mu given reference eta_0
 
    Parameters
    ----------
    eta_0 : reference point. Column vector (x, y, z), normalized.
        Numpy float32 array

    mu : tangent space point. Column vector (x, y, z).
        Numpy float32 array

    Returns
    -------
    eta : coordinate. Column vector (x, y, z), normalized.
        Numpy float32 array

    Raises
    ------
    ValueError : if either eta_0 or mu are not column vectors.
    """

    cdef eigen.Vector3f eta_0_eigen = eigen.Vector3f(eta_0)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f(mu)
    cdef eigen.Vector3f eta_eigen = eigen.Vector3f()

    eta_eigen.v = muToEta_cordal_cpp(eta_0_eigen.v, mu_eigen.v)
    return eta_eigen.toNumpy()


###########################################################
# ROTATION MATRICES
###########################################################

def rotationMatrix(np.ndarray[np.float32_t, ndim=2, mode='c'] eta_0,
                   np.ndarray[np.float32_t, ndim=2, mode='c'] eta_1):
    """
    Returns the rotation matrix between vectors eta_0 and eta_1.

    Parameters
    ----------
    eta_0 : first vector. Column vector (x, y, z), normalized.
        Numpy float32 array

    eta_0 : second vector. Column vector (x, y, z), normalized.
        Numpy float32 array

    Returns
    -------
    R : 3x3 rotation matrix.
        Numpy float32 array

    Raises
    ------
    ValueError : if either eta_0 or eta_1 are not column vectors.
    """
    
    cdef eigen.Vector3f eta_0_eigen = eigen.Vector3f(eta_0)
    cdef eigen.Vector3f eta_1_eigen = eigen.Vector3f(eta_1)

    cdef eigen.Matrix3f R_eigen = eigen.Matrix3f()
    R_eigen.M = rotationMatrix_cpp(eta_0_eigen.v, eta_1_eigen.v)
    return R_eigen.toNumpy()


def rotationMatrixAxisAngle(np.ndarray[np.float32_t, ndim=2, mode='c'] axis,
                            float theta):
    """
    Returns the rotation matrix given the rotation axis and angle.

    Parameters
    ----------
    axis : rotation axis. Column vector (x, y, z), normalized.
        Numpy float32 array

    theta : rotation angle in radians.
        Float32

    Returns
    -------
    R : 3x3 rotation matrix.
        Numpy float32 array

    Raises
    ------
    ValueError : if axis is not a column vector.
    """
    cdef eigen.Vector3f axis_eigen = eigen.Vector3f(axis)

    cdef eigen.Matrix3f R_eigen = eigen.Matrix3f()
    R_eigen.M = rotationMatrixAxisAngle_cpp(axis_eigen.v, theta)
    return R_eigen.toNumpy()


##################################################
# ORTHONORMAL BASIS
##################################################

def getOrthonormalBasis(np.ndarray[np.float32_t, ndim=3, mode='c'] etaGrid,
    int row, int col):
    """
    Returns the orthonormal basis of pixel (row, col)

    Parameters
    ----------
    etaGrid : grid of spherical coordinates. (x, y, z) normalized
        Numpy float32 array

    row : row coordinate

    col : column coordinate

    Returns
    -------
    B : orthonormal basis matrix 2x3
        Numpy float32 array

    norm : norm of B[0,:] vector before normalization
    """
    
    shape = (etaGrid.shape[0], etaGrid.shape[1], etaGrid.shape[2])
    
    # wraps etaGrid array into a PyImageF object
    cdef image.Image_float32 etasWrapped = image.wrap(etaGrid)

    cdef eigen.Matrix2x3f basis_eigen = eigen.Matrix2x3f()
    cdef float norm_out
    basis_eigen.M = getOrthonormalBasis_cpp(etasWrapped.img, row, col, norm_out)
    return basis_eigen.toNumpy(), norm_out


def muToBetapix(np.ndarray[np.float32_t, ndim=2, mode='c'] B, float Bnorm,
    np.ndarray[np.float32_t, ndim=2, mode='c'] mu):

    cdef eigen.Matrix2x3f B_eigen = eigen.Matrix2x3f(B)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f(mu)
    cdef eigen.Vector2f beta_eigen = eigen.Vector2f()

    beta_eigen.v = muToBetapix_cpp(B_eigen.M, Bnorm, mu_eigen.v)
    return beta_eigen.toNumpy()


def betapixToMu(np.ndarray[np.float32_t, ndim=2, mode='c'] B, float Bnorm,
    np.ndarray[np.float32_t, ndim=2, mode='c'] beta):
    
    cdef eigen.Matrix2x3f B_eigen = eigen.Matrix2x3f(B)
    cdef eigen.Vector2f beta_eigen = eigen.Vector2f(beta)
    cdef eigen.Vector3f mu_eigen = eigen.Vector3f()

    mu_eigen.v = betapixToMu_cpp(B_eigen.M, Bnorm, beta_eigen.v)
    return mu_eigen.toNumpy()
    

def betapixFieldToMu(np.ndarray[np.float32_t, ndim=3, mode='c'] etas,
                  np.ndarray[np.float32_t, ndim=3, mode='c'] field,
                  np.ndarray[np.float32_t, ndim=3, mode='c'] output = None):
    
    if output == None:
        output = np.zeros((field.shape[0], field.shape[1], 3), dtype=np.float32)

    # wrap input arrays
    cdef image.Image_float32 etas_wrapped = image.wrap(etas)
    cdef image.Image_float32 field_wrapped = image.wrap(field)
    cdef image.Image_float32 output_wrapped = image.wrap(output)

    betapixFieldToMu_cpp(etas_wrapped.img, field_wrapped.img, output_wrapped.img)

    return output


def muFieldToBetapix(np.ndarray[np.float32_t, ndim=3, mode='c'] etas,
                    np.ndarray[np.float32_t, ndim=3, mode='c'] field,
                    np.ndarray[np.float32_t, ndim=3, mode='c'] output = None):
    
    if output == None:
        output = np.zeros((field.shape[0], field.shape[1], 2), dtype=np.float32)

    # wrap input arrays
    cdef image.Image_float32 etas_wrapped = image.wrap(etas)
    cdef image.Image_float32 field_wrapped = image.wrap(field)
    cdef image.Image_float32 output_wrapped = image.wrap(output)

    muFieldToBetapix_cpp(etas_wrapped.img, field_wrapped.img, output_wrapped.img)

    return output


##################################################
# COORDINATES INTERPOLATION
##################################################

def findInterpolationCoodinates(np.ndarray[np.float32_t, ndim=2, mode='c'] eta,
                                np.ndarray[np.float32_t, ndim=3, mode='c'] etaGrid,
                                flipVertical=True):

    cdef image.Image_float32 etasWrapped = image.wrap(etaGrid)
    cdef eigen.Vector3f eta_eigen = eigen.Vector3f(eta)

    cdef eigen.Vector2f beta = eigen.Vector2f()
    beta.v = findInterpolationCoodinates_cpp(eta_eigen.v, etasWrapped.img, flipVertical);

    return beta.toNumpy()


##################################################
# UTILITY FUNCTIONS
##################################################

def retractField_orthographic(np.ndarray[np.float32_t, ndim=3, mode='c'] etas,
                              np.ndarray[np.float32_t, ndim=3, mode='c'] field,
                              np.ndarray[np.float32_t, ndim=3, mode='c'] output = None):
    
    if output == None:
        output = np.zeros_like(field)

    # wrap input arrays
    cdef image.Image_float32 etas_wrapped = image.wrap(etas)
    cdef image.Image_float32 field_wrapped = image.wrap(field)
    cdef image.Image_float32 output_wrapped = image.wrap(output)

    retractField_orthographic_cpp(etas_wrapped.img, field_wrapped.img, output_wrapped.img)

    return output


def  transformCoordinates(np.ndarray[np.float32_t, ndim=3, mode='c'] field,
                          np.ndarray[np.float32_t, ndim=2, mode='c'] T,
                          np.ndarray[np.float32_t, ndim=3, mode='c'] output = None):
    
    if output == None:
        output = np.zeros_like(field)

    cdef image.Image_float32 field_wrapped = image.wrap(field)
    cdef image.Image_float32 output_wrapped = image.wrap(output)
    cdef eigen.Matrix3f T_eigen = eigen.Matrix3f(T)

    transformCoordinates_cpp(field_wrapped.img, T_eigen.M, output_wrapped.img)

    return output
    

def dotProductField3(np.ndarray[np.float32_t, ndim=3, mode='c'] field1,
                     np.ndarray[np.float32_t, ndim=3, mode='c'] field2,
                     np.ndarray[np.float32_t, ndim=2, mode='c'] output = None):
    """Computes the dot product between two 3D vector fields.

    Parameters
    ----------
    field1: ndarray
        First vector field.

    field2 : ndarray
        Second vector field.

    output : ndarray, optional.
        Output array. If None, it will be created 
        according to the shape of field1.

    Returns
    -------
    output : ndarray.
        Scalar field with the dot product between field1 and field2.
    """
    
    if output == None:
        output = np.zeros((field1.shape[0], field1.shape[1]), dtype=np.float32)

    # wrap input arrays
    cdef image.Image_float32 field1_wrapped = image.wrap(field1)
    cdef image.Image_float32 field2_wrapped = image.wrap(field2)
    cdef image.Image_float32 output_wrapped = image.wrap(output)

    dotProductField3_cpp(field1_wrapped.img, field2_wrapped.img, output_wrapped.img)

    return output


def crossProductField3(np.ndarray[np.float32_t, ndim=3, mode='c'] field1,
                     np.ndarray[np.float32_t, ndim=3, mode='c'] field2,
                     np.ndarray[np.float32_t, ndim=3, mode='c'] output = None):
    """Computes the cross product between two 3D vector fields.

    Parameters
    ----------
    field1: ndarray
        First vector field.

    field2 : ndarray
        Second vector field.

    output : ndarray, optional.
        Output array. If None, it will be created 
        according to the shape of field1.

    Returns
    -------
    output : ndarray.
        Vector field with the cross product between field1 and field2.
    """

    if output == None:
        output = np.zeros((field1.shape[0], field1.shape[1], 3), dtype=np.float32)

    # wrap input arrays
    cdef image.Image_float32 field1_wrapped = image.wrap(field1)
    cdef image.Image_float32 field2_wrapped = image.wrap(field2)
    cdef image.Image_float32 output_wrapped = image.wrap(output)

    crossProductField3_cpp(field1_wrapped.img, field2_wrapped.img, output_wrapped.img)

    return output


def angleBetweenNeighbors(np.ndarray[np.float32_t, ndim=3, mode='c'] etas,
                          np.ndarray[np.float32_t, ndim=2, mode='c'] output = None):
    
    if output == None:
        output = np.zeros((etas.shape[0], etas.shape[1]), dtype=np.float32)

    # wrap input arrays
    cdef image.Image_float32 etas_wrapped = image.wrap(etas)
    cdef image.Image_float32 output_wrapped = image.wrap(output)

    angleBetweenNeighbors_cpp(etas_wrapped.img, output_wrapped.img)

    return output


def distanceBetweenNeighbors(np.ndarray[np.float32_t, ndim=3, mode='c'] etas,
                             np.ndarray[np.float32_t, ndim=2, mode='c'] output = None):
    
    if output == None:
        output = np.zeros((etas.shape[0], etas.shape[1]), dtype=np.float32)

    # wrap input arrays
    cdef image.Image_float32 etas_wrapped = image.wrap(etas)
    cdef image.Image_float32 output_wrapped = image.wrap(output)

    distanceBetweenNeighbors_cpp(etas_wrapped.img, output_wrapped.img)

    return output