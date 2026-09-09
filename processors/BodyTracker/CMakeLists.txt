cmake_minimum_required( VERSION 3.5 )   
project( BodyTracker ) 
 
 
add_library(BodyTracker SHARED BodyTracker.h BodyTracker.c
            ../../tools/ImageOperations/imageOps.c
            ../../tools/ImageOperations/imageOps.h
           ) 
target_link_libraries(BodyTracker m rt ) 
 

set_target_properties(BodyTracker PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
