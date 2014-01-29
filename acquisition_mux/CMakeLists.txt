project( AcquistionMux ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
 
add_library(AcquistionMux SHARED AcquisitionMux.c AcquisitionMux.h ../tools/ImageOperations/imageOps.c  ../tools/ImageOperations/imageOps.h ) 
#target_link_libraries(AcquistionMux rt dl m Timers CalibrationLibrary OperatingSystem ) 
 
 

set_target_properties(AcquistionMux PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
