project( OperatingSystem ) 
cmake_minimum_required( VERSION 2.8.7 )  

add_library(OperatingSystem STATIC main.c OperatingSystem.h  ) 
#target_link_libraries(OperatingSystem rt ) 
 

set_target_properties(OperatingSystem PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
