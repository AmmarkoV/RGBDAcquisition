project( OGLRendererSandbox ) 
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

 


add_definitions(-DINCLUDE_OPENGL_CODE)

add_library(OGLRendererSandbox SHARED  
            glx.c
            glx.h
            main.c
            OGLRendererSandbox.h
            ModelLoader/model_loader.c
            ModelLoader/model_loader.h
            ModelLoader/model_loader_obj.c
            ModelLoader/model_loader_obj.h
            ModelLoader/model_loader_tri.c
            ModelLoader/model_loader_tri.h
            ModelLoader/model_loader_hardcoded.c
            ModelLoader/model_loader_hardcoded.h 
            ModelLoader/model_loader_transform_joints.c
            ModelLoader/model_loader_transform_joints.h
            save_to_file.c
            save_to_file.h
            scene.c
            scene.h
            shader_loader.c
            shader_loader.h
            tiledRenderer.c
            tiledRenderer.h
            tools.c
            tools.h 
            #AmMatrix dependencies 
            ../../../tools/AmMatrix/quaternions.c
            ../../../tools/AmMatrix/quaternions.h
            ../../../tools/AmMatrix/matrix3x3Tools.c
            ../../../tools/AmMatrix/matrix3x3Tools.h
            ../../../tools/AmMatrix/matrix4x4Tools.c
            ../../../tools/AmMatrix/matrix4x4Tools.h
            ../../../tools/AmMatrix/matrixCalculations.c
            ../../../tools/AmMatrix/matrixCalculations.h
            ../../../tools/AmMatrix/solveLinearSystemGJ.c
            ../../../tools/AmMatrix/solveLinearSystemGJ.h
            #ImageOperations dependencies
            ../../../tools/ImageOperations/depthClassifier.c
            ../../../tools/ImageOperations/convolutionFilter.c
            ../../../tools/ImageOperations/imageFilters.c
            ../../../tools/ImageOperations/findSubImage.c
            ../../../tools/ImageOperations/imageOps.c   
            ../../../tools/ImageOperations/patchComparison.c
            ../../../tools/ImageOperations/patchComparison.h 
            ../../../tools/ImageOperations/resize.c 
            #Rest of the stuff  
            TextureLoader/texture_loader.c
            TextureLoader/texture_loader.h
            TextureLoader/image_proc.c
            TextureLoader/image_proc.h
            TrajectoryParser/InputParser_C.c
            TrajectoryParser/InputParser_C.h
            TrajectoryParser/TrajectoryParser.c
            TrajectoryParser/TrajectoryParser.h 
            TrajectoryParser/TrajectoryParserDataStructures.c
            TrajectoryParser/TrajectoryParserDataStructures.h 
            TrajectoryParser/TrajectoryCalculator.c
            TrajectoryParser/TrajectoryCalculator.h 
            TrajectoryParser/TrajectoryPrimitives.c 
            TrajectoryParser/TrajectoryPrimitives.h
            Interfaces/webInterface.c 
            Interfaces/webInterface.h
           ) 
target_link_libraries(OGLRendererSandbox m GL GLU X11 Codecs  ${PNG_Libs} ${JPG_Libs} ) 
 
 

set_target_properties(OGLRendererSandbox PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )
