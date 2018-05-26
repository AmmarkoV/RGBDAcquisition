
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_20,code=sm_20
    )

# Specify include directories
#include_directories( kernels utility )

# Specify library paths
#link_directories( /opt/foobar/lib /opt/joestuff/lib )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    stereoDisparity
    stereoDisparity.cu
    stereoDisparity_kernel.cuh
    )

# For linking ...
# Specify target & libraries to link it with
#target_link_libraries( gpuCompare -lfoobar -ljoestuff )
