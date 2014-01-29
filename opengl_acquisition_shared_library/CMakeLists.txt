project( OpenGLAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
  
add_library(OpenGLAcquisition SHARED 
            OpenGLAcquisition.cpp
            OpenGLAcquisition.h
            opengl_depth_and_color_renderer/src/glx.c
            opengl_depth_and_color_renderer/src/glx.h
            opengl_depth_and_color_renderer/src/main.c
            opengl_depth_and_color_renderer/src/OGLRendererSandbox.h
            opengl_depth_and_color_renderer/src/model_loader.c
            opengl_depth_and_color_renderer/src/model_loader.h
            opengl_depth_and_color_renderer/src/model_loader_obj.c
            opengl_depth_and_color_renderer/src/model_loader_obj.h
            opengl_depth_and_color_renderer/src/save_to_file.c
            opengl_depth_and_color_renderer/src/save_to_file.h
            opengl_depth_and_color_renderer/src/scene.c
            opengl_depth_and_color_renderer/src/scene.h
            opengl_depth_and_color_renderer/src/shader_loader.c
            opengl_depth_and_color_renderer/src/shader_loader.h
            opengl_depth_and_color_renderer/src/tools.c
            opengl_depth_and_color_renderer/src/tools.h
            opengl_depth_and_color_renderer/src/AmMatrix/matrix3x3Tools.c
            opengl_depth_and_color_renderer/src/AmMatrix/matrix3x3Tools.h
            opengl_depth_and_color_renderer/src/AmMatrix/matrix4x4Tools.c
            opengl_depth_and_color_renderer/src/AmMatrix/matrix4x4Tools.h
            opengl_depth_and_color_renderer/src/AmMatrix/matrixCalculations.c
            opengl_depth_and_color_renderer/src/AmMatrix/matrixCalculations.h
            opengl_depth_and_color_renderer/src/AmMatrix/solveLinearSystemGJ.c
            opengl_depth_and_color_renderer/src/AmMatrix/solveLinearSystemGJ.h
            opengl_depth_and_color_renderer/src/TextureLoader/bmp.c
            opengl_depth_and_color_renderer/src/TextureLoader/bmp.h
            opengl_depth_and_color_renderer/src/TextureLoader/ppm.c
            opengl_depth_and_color_renderer/src/TextureLoader/ppm.h
            opengl_depth_and_color_renderer/src/TextureLoader/texture_loader.c
            opengl_depth_and_color_renderer/src/TextureLoader/texture_loader.h
            opengl_depth_and_color_renderer/src/TrajectoryParser/InputParser_C.cpp
            opengl_depth_and_color_renderer/src/TrajectoryParser/InputParser_C.h
            opengl_depth_and_color_renderer/src/TrajectoryParser/TrajectoryParser.cpp
            opengl_depth_and_color_renderer/src/TrajectoryParser/TrajectoryParser.h 
           ) 
target_link_libraries(OpenGLAcquisition rt m CalibrationLibrary GL GLU X11 ) 
 
 

set_target_properties(OpenGLAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
