project( OpenCLRenderer )
cmake_minimum_required( VERSION 2.8.7 )
 
#Make fast, lean and platform independent binaries..
set(CMAKE_CXX_FLAGS "-s -O3 -fPIC -march=native -mtune=native")
set(CMAKE_C_FLAGS "-s -O3 -fPIC -march=native -mtune=native") 

OPTION(OPENCL_OPTIMIZATIONS OFF)

if (OPENCL_OPTIMIZATIONS)
add_definitions(-DOPENCL_OPTIMIZATIONS)
endif(OPENCL_OPTIMIZATIONS)

set_source_files_properties(glProjectf.cl PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(matrixMultiplication.cl PROPERTIES HEADER_FILE_ONLY TRUE)
set_source_files_properties(transform3DPoint.cl PROPERTIES HEADER_FILE_ONLY TRUE)
#add_library(OpenCLRenderer STATIC mat_mul.c)
add_executable(
               OpenCLRenderer 
               openCLRenderer.c
               openCLTools.c
              )  
target_link_libraries(OpenCLRenderer m rt OpenCL pthread )


set_target_properties(OpenCLRenderer PROPERTIES
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
