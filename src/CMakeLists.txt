add_sources(
    arithmetic.cpp
    image.cpp
    geometry.cpp
    springdynamics.cpp
    pixelation.cpp
    camera.cpp
)

# process CMakeLists.txt in gpu folder
add_subdirectory(gpu)

# propagate GPU_SRCS to top level
set (GPU_SRCS ${GPU_SRCS} PARENT_SCOPE)