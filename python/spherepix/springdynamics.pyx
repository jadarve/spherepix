from springdynamics cimport springSystemTimeStep_cpp
from springdynamics cimport runSpringSystem_cpp

import numpy as np
cimport numpy as np

cimport image
import image

def springSystemTimeStep(np.ndarray[np.float32_t, ndim=3, mode='c'] etas_in,
                         np.ndarray[np.float32_t, ndim=3, mode='c'] etasVelocity_in,
                         np.ndarray[np.float32_t, ndim=3, mode='c'] etasAcceleration_in,
                         float dt, float M, float C, float K, float L):
    """
    Runs one time step iteration of the spring system
 
    Parameters
    ----------
    etas_in : input spherical coordinates grid, (x, y, z) normalized
        Numpy float32 array of shape [S, S, 3].
    
    etasVelocity_in : input velocity grid. (vx, vy, vz) lying 
        on the tangent space of etas_in.
        Numpy float32 array of shape [S, S, 3].

    etasAcceleration_in input acceleration grid. (ax, ay, az) lying
        on the tangent space of etas_in.
        Numpy float32 array of shape [S, S, 3].
    
    dt : time step. Float.

    M : point mass. Float.

    C : damping coefficient. Float

    K : spring elasticity constant. Float.

    L : spring rest longitude. Float.

    Returns
    -------
    etas_out : output spherical coordinates grid, (x, y, z) normalized.
        Numpy float32 array of shape [S, S, 3].
 
    etasVelocity_out output velocity grid. (vx, vy, vz) lying 
        on the tangent space of etas_out.
        Numpy float32 array of shape [S, S, 3].
    
    etasAcceleration_out output acceleration grid. (ax, ay, az) lying
       on the tangent space of etas_out.
       Numpy float32 array of shape [S, S, 3]
    """

    shape = (etas_in.shape[0], etas_in.shape[1], etas_in.shape[2])

    # wrap the input arrays in PyImageF objects
    cdef image.Image_float32 etas_in_wrapped = image.copy(etas_in)
    cdef image.Image_float32 etasVelocity_in_wrapped = image.copy(etasVelocity_in)
    cdef image.Image_float32 etasAcceleration_in_wrapped = image.copy(etasAcceleration_in)

    # output arrays
    cdef image.Image_float32 etas_out = image.Image_float32(shape)
    cdef image.Image_float32 etasVelocity_out = image.Image_float32(shape)
    cdef image.Image_float32 etasAcceleration_out = image.Image_float32(shape)


    # call springSystemTimeStep_cpp
    springSystemTimeStep_cpp(etas_in_wrapped.img,
                             etasVelocity_in_wrapped.img,
                             etasAcceleration_in_wrapped.img,
                             etas_out.img,
                             etasVelocity_out.img,
                             etasAcceleration_out.img,
                             dt, M, C, K, L)

    # return
    return etas_out.toNumpy(), etasVelocity_out.toNumpy(), etasAcceleration_out.toNumpy()


def runSpringSystem(np.ndarray[np.float32_t, ndim=3, mode='c'] etas_in,
                    np.ndarray[np.float32_t, ndim=3, mode='c'] etasVelocity_in,
                    np.ndarray[np.float32_t, ndim=3, mode='c'] etasAcceleration_in,
                    float dt, float M, float C, float K, float L, int N):
    """
    Runs the spring system for several iterations.
 
    Parameters
    ----------
    etas_in : input spherical coordinates grid, (x, y, z) normalized
        Numpy float32 array of shape [S, S, 3].
    
    etasVelocity_in : input velocity grid. (vx, vy, vz) lying 
        on the tangent space of etas_in.
        Numpy float32 array of shape [S, S, 3].

    etasAcceleration_in input acceleration grid. (ax, ay, az) lying
        on the tangent space of etas_in.
        Numpy float32 array of shape [S, S, 3].
    
    dt : time step. Float.

    M : point mass. Float.

    C : damping coefficient. Float

    K : spring elasticity constant. Float.

    L : spring rest longitude. Float.

    N : number of iterations

    Returns
    -------
    etas_out : output spherical coordinates grid, (x, y, z) normalized.
        Numpy float32 array of shape [S, S, 3].
 
    etasVelocity_out output velocity grid. (vx, vy, vz) lying 
        on the tangent space of etas_out.
        Numpy float32 array of shape [S, S, 3].
    
    etasAcceleration_out output acceleration grid. (ax, ay, az) lying
       on the tangent space of etas_out.
       Numpy float32 array of shape [S, S, 3]
    """

    shape = (etas_in.shape[0], etas_in.shape[1], etas_in.shape[2])

    # wrap the input arrays in PyImageF objects
    cdef image.Image_float32 etas_in_wrapped = image.copy(etas_in)
    cdef image.Image_float32 etasVelocity_in_wrapped = image.copy(etasVelocity_in)
    cdef image.Image_float32 etasAcceleration_in_wrapped = image.copy(etasAcceleration_in)

    # output arrays
    cdef image.Image_float32 etas_out = image.Image_float32(shape)
    cdef image.Image_float32 etasVelocity_out = image.Image_float32(shape)
    cdef image.Image_float32 etasAcceleration_out = image.Image_float32(shape)

    # call runSpringSystem_cpp
    runSpringSystem_cpp(etas_in_wrapped.img,
                        etasVelocity_in_wrapped.img,
                        etasAcceleration_in_wrapped.img,
                        etas_out.img,
                        etasVelocity_out.img,
                        etasAcceleration_out.img,
                        dt, M, C, K, L, N)

    # return
    return etas_out.toNumpy(), etasVelocity_out.toNumpy(), etasAcceleration_out.toNumpy()