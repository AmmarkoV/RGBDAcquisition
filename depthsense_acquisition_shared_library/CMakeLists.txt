project( DepthSenseAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
    

add_library(DepthSenseAcquisition SHARED 
            DepthSenseAcquisition.c
            DepthSenseAcquisition.h 
           ) 
target_link_libraries(DepthSenseAcquisition rt m CalibrationLibrary OperatingSystem Codecs ) 
 
 

set_target_properties(DepthSenseAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
