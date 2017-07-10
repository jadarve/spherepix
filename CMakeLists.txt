cmake_minimum_required(VERSION 2.8.11)
project(spherepix)


#################################################
# MACROS
#################################################

macro (add_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        if (_relPath)
            list (APPEND SRCS "${_relPath}/${_src}")
        else()
            list (APPEND SRCS "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (SRCS ${SRCS} PARENT_SCOPE)
    endif()
endmacro()


macro (add_gpu_sources)
    file (RELATIVE_PATH _relPath "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}")
    foreach (_src ${ARGN})
        # message(STATUS "source: " ${_src})
        if (_relPath)
            list (APPEND GPU_SRCS "${_relPath}/${_src}")
        else()
            list (APPEND GPU_SRCS "${_src}")
        endif()
    endforeach()
    if (_relPath)
        # propagate SRCS to parent directory
        set (GPU_SRCS ${GPU_SRCS} PARENT_SCOPE)
    endif()
    # message(STATUS "gpu sources: " ${GPU_SRCS})
endmacro()



#################################################
# LIBRARY SETTINGS
#################################################
set (LIBRARY_OUTPUT_PATH   "lib")

# assumes flowfilter_gpu
# are installed at /usr/local/lib
set( LIBS
    flowfilter_gpu
)

#################################################
# INCLUDE SETTINGS
#################################################
include_directories(
    "${PROJECT_SOURCE_DIR}/include"
    /usr/include/eigen3
)


#################################################
# COMPILER SETTINGS
#################################################
set (CMAKE_CXX_COMPILER         "g++")
# set (CMAKE_CXX_COMPILER         "clang++")

# -flto = link time optimization
# -fopenmp = use OpenMP
set (CMAKE_CXX_FLAGS            "-std=c++11 -flto -fopenmp -O3 -Wall")
# set (CMAKE_CXX_FLAGS            "-std=c++11 -Wall")


#################################################
# SOURCE FILES SETTINGS
#################################################
# process the CMakeLists.txt file at src/ folder
# it should add all the source files in the folder
add_subdirectory(src)


#################################################
# CUDA SETTINGS
#################################################
find_package(CUDA)
if (CUDA_FOUND)
    
    message(STATUS "found CUDA")

    # NVCC compiler flags
    set(CUDA_NVCC_FLAGS             "-std=c++11 -O3")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)    

endif(CUDA_FOUND)


#################################################
# TARGETS
#################################################
message(STATUS "C++ sources: ${SRCS}")
message(STATUS "GPU sources: " ${GPU_SRCS})

# spherepix CPU
add_library(spherepix SHARED ${SRCS})


if(CUDA_FOUND)
    
    # spherepix_gpu library with CUDA implementation
    cuda_add_library(spherepix_gpu SHARED ${GPU_SRCS})
    target_link_libraries(spherepix_gpu spherepix ${LIBS})

endif(CUDA_FOUND)


#################################################
# INSTALL
#################################################

# install
install(
    TARGETS spherepix
    LIBRARY DESTINATION lib
)

if(CUDA_FOUND)
    install(
        TARGETS spherepix_gpu
        LIBRARY DESTINATION lib
    )
endif(CUDA_FOUND)

# install header files
install(
    DIRECTORY include/spherepix
    DESTINATION include
)


#################################################
# TESTING
#################################################
# enable_testing()
# add_executable(imageTest test/main.cpp)
# target_link_libraries(imageTest spherepix)
# add_test(NAME imageTest COMMAND imageTest)

