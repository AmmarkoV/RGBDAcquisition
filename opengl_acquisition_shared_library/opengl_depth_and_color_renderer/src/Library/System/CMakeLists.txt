project( GLXTests ) 
cmake_minimum_required( VERSION 2.8.7 )
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
   

#You need to compile the main RGBDAcquisition project to get these 2 files..
ADD_LIBRARY(RGBDAcquisitionLibrary SHARED IMPORTED) 
SET_TARGET_PROPERTIES(RGBDAcquisitionLibrary PROPERTIES IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../../acquisition/libRGBDAcquisition.so") 

ADD_LIBRARY(CalibrationLibrary STATIC IMPORTED)
SET_TARGET_PROPERTIES(CalibrationLibrary PROPERTIES IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/../../../../../tools/Calibration/libCalibrationLibrary.a")  

add_definitions(-DUSE_GLEW)
 
  
add_executable(gl3 glx_test.c glx3.c ../Rendering/ShaderPipeline/shader_loader.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c)
target_link_libraries(gl3 rt m GL GLU GLEW X11)  
set_target_properties(gl3 PROPERTIES 
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )

add_executable(gl3Tex glx_testRenderToTexture.c glx3.c ../Rendering/ShaderPipeline/shader_loader.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c)
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



add_executable(gl3MultiDiff glx3.c glx_testMultiViewportDiff.c ../Rendering/ShaderPipeline/shader_loader.c ../Rendering/ShaderPipeline/render_buffer.c ../Rendering/ShaderPipeline/uploadGeometry.c ../Rendering/downloadFromRenderer.c ../Rendering/ShaderPipeline/computeShader.c  ../Rendering/ShaderPipeline/uploadTextures.c  ../ModelLoader/model_loader_tri.c ../Tools/save_to_file.c ../Tools/tools.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/quaternions.h  ../../../../../tools/AmMatrix/matrixOpenGL.c )
target_link_libraries(gl3MultiDiff rt m GL GLU GLEW X11 RGBDAcquisitionLibrary CalibrationLibrary)  
set_target_properties(gl3MultiDiff PROPERTIES 
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )

 
add_definitions(-DUSE_GLEW)
add_executable(gl3MeshTransform glx3.c glx_testMeshTransform.c ../Rendering/ShaderPipeline/shader_loader.c ../Rendering/ShaderPipeline/render_buffer.c ../Rendering/ShaderPipeline/uploadGeometry.c ../Rendering/downloadFromRenderer.c  ../Rendering/ShaderPipeline/computeShader.c  ../Rendering/ShaderPipeline/uploadTextures.c ../ModelLoader/model_loader_tri.c ../ModelLoader/model_loader_transform_joints.c ../TrajectoryParser/InputParser_C.c ../Tools/save_to_file.c ../Tools/tools.c ../../../../../tools/AmMatrix/matrix4x4Tools.c ../../../../../tools/AmMatrix/matrixOpenGL.c ../../../../../tools/AmMatrix/simpleRenderer.c ../../../../../tools/AmMatrix/quaternions.c ../MotionCaptureLoader/bvh_loader.c ../MotionCaptureLoader/calculate/bvh_project.c ../MotionCaptureLoader/calculate/bvh_to_tri_pose.c ../MotionCaptureLoader/calculate/bvh_transform.c ../MotionCaptureLoader/edit/bvh_cut_paste.c ../MotionCaptureLoader/edit/bvh_filter.c ../MotionCaptureLoader/edit/bvh_interpolate.c ../MotionCaptureLoader/edit/bvh_merge.c ../MotionCaptureLoader/edit/bvh_randomize.c ../MotionCaptureLoader/edit/bvh_remapangles.c ../MotionCaptureLoader/edit/bvh_rename.c ../MotionCaptureLoader/export/bvh_export.c ../MotionCaptureLoader/export/bvh_to_bvh.c ../MotionCaptureLoader/export/bvh_to_c.c ../MotionCaptureLoader/export/bvh_to_csv.c ../MotionCaptureLoader/export/bvh_to_svg.c ../MotionCaptureLoader/export/bvh_to_trajectoryParserPrimitives.c ../MotionCaptureLoader/export/bvh_to_trajectoryParserTRI.c ../MotionCaptureLoader/ik/bvh_inverseKinematics.c ../MotionCaptureLoader/ik/hardcodedProblems_inverseKinematics.c ../MotionCaptureLoader/ik/levmar.c ../MotionCaptureLoader/import/fromBVH.c ../MotionCaptureLoader/metrics/bvh_measure.c )
target_link_libraries(gl3MeshTransform rt m GL GLU GLEW X11 pthread )  
set_target_properties(gl3MeshTransform PROPERTIES 
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )

