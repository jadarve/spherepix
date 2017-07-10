
cimport image

cdef extern from 'spherepix/springdynamics.h' namespace 'spherepix':
    
    void springSystemTimeStep_cpp 'spherepix::springSystemTimeStep'(
                              const image.Image_cpp[float]& etas_in,
                              const image.Image_cpp[float]& etasVelocity_in,
                              const image.Image_cpp[float]& etasAcceleration_in,
                              image.Image_cpp[float]& etas_out,
                              image.Image_cpp[float]& etasVelocity_out,
                              image.Image_cpp[float]& etasAcceleration_out,
                              const float dt,
                              const float M,
                              const float C,
                              const float K,
                              const float L)


    void runSpringSystem_cpp 'spherepix::runSpringSystem'(
                         const image.Image_cpp[float]& etas_in,
                         const image.Image_cpp[float]& etasVelocity_in,
                         const image.Image_cpp[float]& etasAcceleration_in,
                         image.Image_cpp[float]& etas_out,
                         image.Image_cpp[float]& etasVelocity_out,
                         image.Image_cpp[float]& etasAcceleration_out,
                         const float dt,
                         const float M,
                         const float C,
                         const float K,
                         const float L,
                         const int N)