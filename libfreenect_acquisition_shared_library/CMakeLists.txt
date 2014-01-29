project( FreenectAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
  
    add_library(FreenectAcquisition SHARED FreenectAcquisition.c FreenectAcquisition.h ) 
    target_link_libraries(FreenectAcquisition rt dl m Timers CalibrationLibrary OperatingSystem ) 
 #TODO THIS IS NOT WORKING
 

set_target_properties(FreenectAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
