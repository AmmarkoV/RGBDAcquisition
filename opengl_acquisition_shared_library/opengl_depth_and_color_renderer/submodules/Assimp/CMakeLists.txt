project( assimpLoader ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
  
OPTION (ENABLE_JPG ON)
OPTION (ENABLE_PNG ON)

IF( ENABLE_JPG )
 MESSAGE("JPGs will be included in this codec build")
 set(JPG_Libs jpeg ) 
 add_definitions(-DUSE_JPG_FILES)
 add_definitions(-DENABLE_JPG)
ENDIF( ENABLE_JPG )


IF( ENABLE_PNG )
 MESSAGE("PNGs will be included in this codec build")
 set(PNG_Libs png ) 
 add_definitions(-DUSE_PNG_FILES)
 add_definitions(-DENABLE_PNG)
ENDIF( ENABLE_PNG )

#Add the codecs needed for the correct building
set(CODECS_DIRECTORY "../../../../tools/Codecs") 
add_subdirectory ( ${CODECS_DIRECTORY} RGBDAcquisition/Codecs)

message("Don't foget to sudo apt-get install libassimp-dev")
 
add_library(assimpLoader SHARED  
            assimp_loader.cpp
            assimp_loader.h   
               ../../src/Library/ModelLoader/model_loader_tri.c
               ../../src/Library/ModelLoader/model_loader_tri.h
           ) 
target_link_libraries(assimpLoader m assimp) 
 


add_executable(assimpTester 
               assimp_main.cpp 
               assimp_loader.cpp
               assimp_loader.h  
               assimp_bvh.cpp
               assimp_bvh.h
               ../../src/Library/ModelLoader/model_loader_tri.c
               ../../src/Library/ModelLoader/model_loader_tri.h 
               ) 
target_link_libraries(assimpTester rt m assimp Codecs ${PNG_Libs} ${JPG_Libs} ) 
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
