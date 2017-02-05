project( assimpLoader ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
  
 
 
add_library(assimpLoader SHARED  
            assimp_loader.cpp
            assimp_loader.h   
               ../../src/ModelLoader/model_loader_tri.c
               ../../src/ModelLoader/model_loader_tri.h
           ) 
target_link_libraries(assimpLoader m assimp) 
 


add_executable(assimpTester 
               assimp_main.cpp 
               assimp_loader.cpp
               assimp_loader.h  
               assimp_bvh.cpp
               assimp_bvh.h
               ../../src/ModelLoader/model_loader_tri.c
               ../../src/ModelLoader/model_loader_tri.h 
               ) 
target_link_libraries(assimpTester rt m assimp ) 
#add_dependencies(assimpTester OGLRendererSandbox)  

 

set_target_properties(assimpLoader PROPERTIES 
                       #DEBUG_POSTFIX "D"
                          ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )


set_target_properties(assimpTester PROPERTIES 
                       #DEBUG_POSTFIX "D"
                          ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
