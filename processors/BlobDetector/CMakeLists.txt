cmake_minimum_required( VERSION 3.5 )   
project( BlobDetector ) 
 
add_library(BlobDetector SHARED BlobDetector.h BlobDetector.c
            ../../tools/ImageOperations/imageOps.c
            ../../tools/ImageOperations/imageOps.h
           ) 
target_link_libraries(BlobDetector m rt ) 
 

set_target_properties(BlobDetector PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
