project( fast ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
 
    
  
    add_library(fast STATIC fast.c fast.h nonmax.c fast_9.c fast_10.c fast_11.c fast_12.c ) 
    target_link_libraries(fast m ) 
 
 

set_target_properties(fast PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
