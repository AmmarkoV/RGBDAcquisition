cmake_minimum_required( VERSION 3.5 )   
project( ViewpointChange ) 
 
add_library(ViewpointChange STATIC viewpoint_change.c viewpoint_change.c main.c  ) 
target_link_libraries(ViewpointChange m rt ) 
 

set_target_properties(ViewpointChange PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
