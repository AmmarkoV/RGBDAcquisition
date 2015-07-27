project( OGLRendererSandbox ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
  
add_library(OGLRendererSandbox SHARED  
            glx.c
            glx.h
            main.c
            OGLRendererSandbox.h
            ModelLoader/model_loader.c
            ModelLoader/model_loader.h
            ModelLoader/model_loader_obj.c
            ModelLoader/model_loader_obj.h
            ModelLoader/model_loader_hardcoded.c
            ModelLoader/model_loader_hardcoded.h
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
            TextureLoader/bmp.c
            TextureLoader/bmp.h
            TextureLoader/ppm.c
            TextureLoader/ppm.h
            TextureLoader/texture_loader.c
            TextureLoader/texture_loader.h
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
target_link_libraries(OGLRendererSandbox m GL GLU X11 ) 
 
 

set_target_properties(OGLRendererSandbox PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )
