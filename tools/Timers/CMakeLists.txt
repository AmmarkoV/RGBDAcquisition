cmake_minimum_required( VERSION 3.5 )   
project( Timers ) 
 
add_library(Timers STATIC timer.c timer.h  ) 
target_link_libraries(Timers m rt ) 
 

set_target_properties(Timers PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
