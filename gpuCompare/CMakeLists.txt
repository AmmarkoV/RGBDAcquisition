
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 
    #-gencode arch=compute_20,code=sm_20
    )

# Specify include directories
#include_directories( kernels utility )

# Specify library paths
#link_directories( /opt/foobar/lib /opt/joestuff/lib )

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    gpuCompare
    gpuCompare.cu
    gpuCompare_kernel.cuh
    )


target_link_libraries(gpuCompare rt dl m RGBDAcquisition OperatingSystem Timers CalibrationLibrary  ${LocationService_Libs} )
set_target_properties(gpuCompare PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(gpuCompare Acquisition OperatingSystem Timers CalibrationLibrary) 
set_target_properties(gpuCompare PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
