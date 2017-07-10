

cdef extern from 'Eigen/Dense' namespace 'Eigen':

    cdef cppclass Vector2f_cpp 'Eigen::Vector2f':
        Vector2f_cpp() except +
        float* data()
    
    cdef cppclass Vector3f_cpp 'Eigen::Vector3f':
        Vector3f_cpp() except +
        float* data()
    
    cdef cppclass Matrix2Xf_cpp 'Eigen::Matrix2Xf':
        Matrix2Xf_cpp() except +
        Matrix2Xf_cpp(const int rows, const int cols)
        float* data()
    
    cdef cppclass Matrix3f_cpp 'Eigen::Matrix3f':
        Matrix3f_cpp() except +
        float* data()


#################################################
# WRAPPER CLASSES
#################################################

cdef class Vector2f:
    cdef Vector2f_cpp v
    cdef float* data


cdef class Vector3f:
    cdef Vector3f_cpp v
    cdef float* data


cdef class Matrix3f:
    cdef Matrix3f_cpp M
    cdef float* data


cdef class Matrix2x3f:
    cdef Matrix2Xf_cpp M
    cdef float* data

