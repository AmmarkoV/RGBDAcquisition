project( ModelDump ) 
cmake_minimum_required( VERSION 2.8.7 )
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
   
add_executable(ModelDump modelDump.c ) 
target_link_libraries(ModelDump rt m GL GLU X11 OGLRendererSandbox  ) 
add_dependencies(ModelDump OGLRendererSandbox)  
       
 
set_target_properties(ModelDump PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )
