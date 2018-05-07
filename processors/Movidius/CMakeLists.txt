project( Movidius ) 
cmake_minimum_required( VERSION 2.8.7 )   
 
add_library(Movidius SHARED Movidius.h Movidius.c imageConversions.c fp16.c tinyYolo.c ) 
target_link_libraries(Movidius m rt mvnc) 
 

set_target_properties(Movidius PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
