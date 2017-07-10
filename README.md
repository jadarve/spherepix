# spherepix

A data structure for processing spherical images.

    @Article{2017_Adarve_RAL,
      Title                    = {Spherepix: A Data Structure for Spherical Image Processing},
      Author                   = {J. D. Adarve and R. Mahony},
      Journal                  = {IEEE Robotics and Automation Letters},
      Year                     = {2017}
    }


# Build and Installation

## Dependencies

  * CMake 2.8.11 or higher
  * Cuda 7.5 or above
  * Build and install the optical flow filter library. Available at https://github.com/jadarve/optical-flow-filter

## Build (Linux)

    git clone https://github.com/jadarve/spherepix
    cd spherepix
    mkdir build
    cd build
    cmake ..
    make
    sudo make install 
    

The library and header files will be installed at **/usr/local/lib** and **/usr/local/include** respectively.


## Python Wrappers

A python package with wrappers to the C++ library is available at **spherepix/python/** folder. The wrappers have been developed and build using Cython 0.23.4.

    cd spherepix/python/
    python setup.py build
    sudo python setup.py install
