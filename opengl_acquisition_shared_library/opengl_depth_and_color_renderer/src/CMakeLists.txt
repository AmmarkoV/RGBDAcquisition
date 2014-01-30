project( OGLRendererSandbox ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
  
add_library(OGLRendererSandbox SHARED  
            glx.c
            glx.h
            main.c
            OGLRendererSandbox.h
            model_loader.c
            model_loader.h
            model_loader_obj.c
            model_loader_obj.h
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
            AmMatrix/matrix3x3Tools.c
            AmMatrix/matrix3x3Tools.h
            AmMatrix/matrix4x4Tools.c
            AmMatrix/matrix4x4Tools.h
            AmMatrix/matrixCalculations.c
            AmMatrix/matrixCalculations.h
            AmMatrix/solveLinearSystemGJ.c
            AmMatrix/solveLinearSystemGJ.h
            TextureLoader/bmp.c
            TextureLoader/bmp.h
            TextureLoader/ppm.c
            TextureLoader/ppm.h
            TextureLoader/texture_loader.c
            TextureLoader/texture_loader.h
            TrajectoryParser/InputParser_C.cpp
            TrajectoryParser/InputParser_C.h
            TrajectoryParser/TrajectoryParser.cpp
            TrajectoryParser/TrajectoryParser.h 
           ) 
target_link_libraries(OGLRendererSandbox m GL GLU X11 ) 
 
 

set_target_properties(OGLRendererSandbox PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )
