project( V4L2Acquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  

add_library(V4L2Acquisition SHARED 
            PixelFormatConversions.c
            PixelFormatConversions.h
            V4L2_c.c
            V4L2_c.h
            V4L2Acquisition.c
            V4L2Acquisition.h
            V4L2IntrinsicCalibration.c
            V4L2IntrinsicCalibration.h
            V4L2Wrapper.c
            V4L2Wrapper.h 
           ) 
target_link_libraries(V4L2Acquisition rt CalibrationLibrary ) 
 
 

set_target_properties(V4L2Acquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
