project( ViewerOGL ) 
cmake_minimum_required( VERSION 2.8.7 )



add_executable(ViewerOGL 
               main.c 
               ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/System/glx3.c  
               ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/Tools/tools.c  
               ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/shader_loader.c
               ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/uploadGeometry.c 
               ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/render_buffer.c 
               ../tools/AmMatrix/matrix4x4Tools.c
               ../tools/AmMatrix/matrixOpenGL.c
              )  

target_link_libraries(ViewerOGL rt dl m RGBDAcquisition OperatingSystem Timers CalibrationLibrary ${LocationService_Libs} GL GLU GLEW X11 )
set_target_properties(ViewerOGL PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(ViewerOGL Acquisition OperatingSystem Timers CalibrationLibrary) 
set_target_properties(ViewerOGL PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
