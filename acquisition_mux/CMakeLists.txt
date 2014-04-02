project( AcquisitionMux ) 
cmake_minimum_required( VERSION 2.8.7 )   
 
add_library(AcquisitionMux STATIC AcquisitionMux.c AcquisitionMux.h ../tools/ImageOperations/imageOps.c ../tools/ImageOperations/imageOps.h ) 
#target_link_libraries(AcquisitionMux rt dl m Timers CalibrationLibrary OperatingSystem ) 
 
 

set_target_properties(AcquisitionMux PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
