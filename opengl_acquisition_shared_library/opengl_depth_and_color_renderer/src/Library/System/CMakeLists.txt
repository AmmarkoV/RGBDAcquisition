project( GLXTests ) 
cmake_minimum_required( VERSION 2.8.7 )
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
   

add_definitions(-DUSE_GLEW)
 
  
add_executable(gl3 glx_test.c glx3.c ../Rendering/ShaderPipeline/shader_loader.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c)
target_link_libraries(gl3 rt m GL GLU GLEW X11)  
set_target_properties(gl3 PROPERTIES 
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )

add_executable(gl3Tex glx_test.c glx3.c ../Rendering/ShaderPipeline/shader_loader.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c)
target_link_libraries(gl3Tex rt m GL GLU GLEW X11)  
set_target_properties(gl3Tex PROPERTIES 
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )


add_executable(gl3Multi glx3.c glx_testMultiViewport.c ../Rendering/ShaderPipeline/shader_loader.c ../Rendering/ShaderPipeline/render_buffer.c ../Rendering/ShaderPipeline/uploadGeometry.c ../Rendering/downloadFromRenderer.c  ../Rendering/ShaderPipeline/computeShader.c  ../Rendering/ShaderPipeline/uploadTextures.c ../ModelLoader/model_loader_tri.c ../Tools/save_to_file.c ../Tools/tools.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c )
target_link_libraries(gl3Multi rt m GL GLU GLEW X11)  
set_target_properties(gl3Multi PROPERTIES 
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )

