cimport eigen

from libc.string cimport memcpy

import cython

cimport numpy as np
import numpy as np


cdef class Vector2f:

    def __cinit__(self, np.ndarray[np.float32_t, ndim=2, mode='c'] arr = None):

        self.data = self.v.data()

        if arr != None:
            # copy the data into self.v
            self.fromNumpy(arr)

    def toNumpy(self):
        """
        Creates a numpy 3-column vector with a copy of the data
        """

        cdef np.ndarray out = np.zeros((2,1), dtype=np.float32)
        memcpy(<void*>out.data, <void*>self.data, 2*cython.sizeof(cython.float))
        return out


    def fromNumpy(self, np.ndarray[np.float32_t, ndim=2, mode='c'] arr):
        """
        Copy the content of a 3-column numpy vector to this object
        """

        if not (arr.shape[0] == 2 and arr.shape[1] == 1):
            raise ValueError(
                'wrong numpy array shape : ({0}, {1}). It should be (2,1)'.format(
                    arr.shape[0], arr.shape[1]))

        memcpy(<void*>self.data, <void*>arr.data, 2*cython.sizeof(cython.float))


cdef class Vector3f:

    def __cinit__(self, np.ndarray[np.float32_t, ndim=2, mode='c'] arr = None):

        self.data = self.v.data()

        if arr != None:
            # copy the data into self.v
            self.fromNumpy(arr)

    def toNumpy(self):
        """
        Creates a numpy 3-column vector with a copy of the data
        """

        cdef np.ndarray out = np.zeros((3,1), dtype=np.float32)
        memcpy(<void*>out.data, <void*>self.data, 3*cython.sizeof(cython.float))
        return out


    def fromNumpy(self, np.ndarray[np.float32_t, ndim=2, mode='c'] arr):
        """
        Copy the content of a 3-column numpy vector to this object
        """

        if not (arr.shape[0] == 3 and arr.shape[1] == 1):
            raise ValueError(
                'wrong numpy array shape : ({0}, {1}). It should be (3,1)'.format(
                    arr.shape[0], arr.shape[1]))

        memcpy(<void*>self.data, <void*>arr.data, 3*cython.sizeof(cython.float))


cdef class Matrix3f:
    
    def __cinit__(self, np.ndarray[np.float32_t, ndim=2, mode='c'] arr = None):

        self.data = self.M.data()

        if arr != None:

            # copy the data into self.M
            self.fromNumpy(arr)


    def toNumpy(self):
        """
        Creates a numpy copy of the matrix
        """
        
        # creates a column-major matrix to copy the data from
        # Eigen matrix (also in column-major order)
        cdef np.ndarray out = np.zeros((3,3), dtype=np.float32, order='F')
        memcpy(<void*>out.data, <void*>self.data, 9*cython.sizeof(cython.float))

        # return a copy of out in row-major order
        return np.copy(out, order='C')


    def fromNumpy(self, np.ndarray[np.float32_t, ndim=2, mode='c'] arr):
        """
        Copies the values from a numpy array to this matrix
        """

        if not (arr.shape[0] == 3 and arr.shape[1] == 3):
            raise ValueError(
                'wrong numpy array shape: ({0}, {1}). It should be (3, 3)'.format(
                    arr.shape[0], arr.shape[1]))

        # copies the numpy matrix buffer to Eigen buffer
        # this requires the Numpy matrix to be in F order
        cdef np.ndarray arrF = np.copy(arr, order='F')
        memcpy(<void*>self.data, <void*>arrF.data, 9*cython.sizeof(cython.float))


cdef class Matrix2x3f:
    
    def __cinit__(self, np.ndarray[np.float32_t, ndim=2, mode='c'] arr = None):

        self.M = eigen.Matrix2Xf_cpp(2,3)
        self.data = self.M.data()

        if arr != None:

            # copy the data into self.M
            self.fromNumpy(arr)


    def toNumpy(self):
        """
        Creates a numpy copy of the matrix
        """
        
        # creates a column-major matrix to copy the data from
        # Eigen matrix (also in column-major order)
        cdef np.ndarray out = np.zeros((2,3), dtype=np.float32, order='F')
        memcpy(<void*>out.data, <void*>self.data, 6*cython.sizeof(cython.float))

        # return a copy of out in row-major order
        return np.copy(out, order='C')


    def fromNumpy(self, np.ndarray[np.float32_t, ndim=2, mode='c'] arr):
        """
        Copies the values from a numpy array to this matrix
        """

        if not (arr.shape[0] == 2 and arr.shape[1] == 3):
            raise ValueError(
                'wrong numpy array shape: ({0}, {1}). It should be (2, 3)'.format(
                    arr.shape[0], arr.shape[1]))

        # copies the numpy matrix buffer to Eigen buffer
        # this requires the Numpy matrix to be in F order
        cdef np.ndarray arrF = np.copy(arr, order='F')
        memcpy(<void*>self.data, <void*>arrF.data, 6*cython.sizeof(cython.float))