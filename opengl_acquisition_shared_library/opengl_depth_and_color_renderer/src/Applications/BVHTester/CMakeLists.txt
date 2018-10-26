project( BVHTester ) 
cmake_minimum_required( VERSION 2.8.7 )
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
   


IF( ENABLE_JPG )
 MESSAGE("JPGs will be included in this codec build")
 set(JPG_Libs jpeg ) 
 set(JPG_Parts jpgInput.c jpgInput.h jpgExifexternal.c jpgExifexternal.h jpgExiforient_embed.c jpgExiforient_embed.h )
 set(JPG_Includes ${CMAKE_SOURCE_DIR}/3dparty/OpenNI2/Include/ )
 add_definitions(-DUSE_JPG_FILES)
 add_definitions(-DENABLE_JPG)
ENDIF( ENABLE_JPG )


IF( ENABLE_PNG )
 MESSAGE("PNGs will be included in this codec build")
 set(PNG_Libs png ) 
 set(PNG_Parts pngInput.c pngInput.h)
 set(PNG_Includes ${CMAKE_SOURCE_DIR}/3dparty/OpenNI2/Include/ )
 add_definitions(-DUSE_PNG_FILES)
 add_definitions(-DENABLE_PNG)
ENDIF( ENABLE_PNG )

IF( ENABLE_SHADERS )
 MESSAGE("Shaders will be included in this codec build")
 set(GLEW_Libs GLEW ) #sudo apt-get install libglew-dev 
 set(GLEW_Parts  )
 set(GLEW_Includes  ) 
 add_definitions(-DUSE_GLEW)
ENDIF( ENABLE_SHADERS )

add_executable( 
               BVHTester 
               main.c 
              ../../Library/MotionCaptureLoader/bvh_loader.c
              ../../Library/MotionCaptureLoader/bvh_to_trajectoryParser.c
              ../../Library/MotionCaptureLoader/bvh_transform.c
              ../../Library/TrajectoryParser/InputParser_C.c
              ../../../../../tools/AmMatrix/matrix4x4Tools.c
              )
 
target_link_libraries(BVHTester rt m  ) 
add_dependencies(BVHTester OGLRendererSandbox)  
       
 
set_target_properties(BVHTester PROPERTIES 
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )
