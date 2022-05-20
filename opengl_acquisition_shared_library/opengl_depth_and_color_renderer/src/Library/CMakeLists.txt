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

 
IF( ENABLE_SHADERS )
 MESSAGE("Shaders will be included in this codec build")
 set(GLEW_Libs GLEW )  #sudo apt-get install libglew-dev
 set(GLEW_Parts  )
 set(GLEW_Includes  ) 
 add_definitions(-DUSE_GLEW)
ENDIF( ENABLE_SHADERS )

 
IF( ENABLE_AMMARSERVER )
 MESSAGE("Shaders will be included in this codec build")
 set(HTTP_Libs AmmarServer )  #sudo apt-get install libglew-dev 
 add_definitions(-DUSE_AMMARSERVER)
ENDIF( ENABLE_AMMARSERVER )

add_definitions(-DINCLUDE_OPENGL_CODE)

add_library(OGLRendererSandbox SHARED 
            System/glx.h
            System/glx.c
            System/glx2.h
            System/glx2.c
            System/glx3.h
            System/glx3.c
            main.c
            OGLRendererSandbox.h
            #3D Models and how to load them   
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
            ModelLoader/tri_bvh_controller.h
            ModelLoader/model_processor.c
            ModelLoader/model_processor.h
            ModelLoader/model_converter.c
            ModelLoader/model_converter.h
            ModelLoader/model_editor.c
            ModelLoader/model_editor.h
            #Motion Capture and how to load it
            MotionCaptureLoader/bvh_loader.c
            MotionCaptureLoader/calculate/bvh_transform.c
            MotionCaptureLoader/calculate/bvh_project.c
            MotionCaptureLoader/calculate/bvh_to_tri_pose.c
            MotionCaptureLoader/calculate/smoothing.h
            MotionCaptureLoader/calculate/smoothing.c
            MotionCaptureLoader/import/fromBVH.c
            MotionCaptureLoader/export/bvh_to_trajectoryParserTRI.c
            MotionCaptureLoader/export/bvh_to_trajectoryParserPrimitives.c
            MotionCaptureLoader/export/bvh_to_c.h 
            MotionCaptureLoader/export/bvh_to_c.c 
            MotionCaptureLoader/export/bvh_to_svg.h
            MotionCaptureLoader/export/bvh_to_svg.c
            MotionCaptureLoader/export/bvh_to_csv.h
            MotionCaptureLoader/export/bvh_to_csv.c
            MotionCaptureLoader/export/bvh_to_json.h
            MotionCaptureLoader/export/bvh_to_json.c
            MotionCaptureLoader/export/bvh_to_bvh.h
            MotionCaptureLoader/export/bvh_to_bvh.c
            MotionCaptureLoader/export/bvh_export.c 
            MotionCaptureLoader/edit/bvh_randomize.c 
            MotionCaptureLoader/edit/bvh_rename.c 
            MotionCaptureLoader/edit/bvh_remapangles.c 
            MotionCaptureLoader/edit/bvh_interpolate.c 
            MotionCaptureLoader/edit/bvh_filter.c 
            MotionCaptureLoader/edit/bvh_merge.c 
            MotionCaptureLoader/edit/bvh_cut_paste.c 
            #Inverse Kinematics
            MotionCaptureLoader/ik/bvh_inverseKinematics.c
            MotionCaptureLoader/ik/hardcodedProblems_inverseKinematics.c
            #MotionCaptureLoader/ik/levmar.c
            MotionCaptureLoader/metrics/bvh_measure.c  
            MotionCaptureLoader/tests/test.c
            #Textures and how to load them  
            TextureLoader/texture_loader.c
            TextureLoader/texture_loader.h
            TextureLoader/image_proc.c
            TextureLoader/image_proc.h
            #OpenGL Rendering stuff  
            Rendering/FixedPipeline/ogl_fixed_pipeline_renderer.c
            Rendering/FixedPipeline/ogl_fixed_pipeline_renderer.h
            Rendering/ShaderPipeline/render_buffer.c
            Rendering/ShaderPipeline/render_buffer.h
            Rendering/ShaderPipeline/uploadGeometry.h
            Rendering/ShaderPipeline/uploadGeometry.c 
            Rendering/ShaderPipeline/ogl_shader_pipeline_renderer.c
            Rendering/ShaderPipeline/ogl_shader_pipeline_renderer.h
            Rendering/ShaderPipeline/shader_loader.c
            Rendering/ShaderPipeline/shader_loader.h
            Rendering/downloadFromRenderer.c
            Rendering/downloadFromRenderer.h
            Rendering/ogl_rendering.c
            Rendering/ogl_rendering.h
            Rendering/tiledRenderer.c
            Rendering/tiledRenderer.h
            Tools/tools.h 
            Tools/tools.c
            Tools/save_to_file.c
            Tools/save_to_file.h
            Scene/scene.h
            Scene/scene.c
            Scene/control.h
            Scene/control.c
            Scene/photoShootingScene.h
            Scene/photoShootingScene.c
            #PThreads automation
            ../../../../tools/PThreadWorkerPool/pthreadWorkerPool.h
            #AmMatrix dependencies 
            ../../../../tools/AmMatrix/matrix3x3Tools.c 
            ../../../../tools/AmMatrix/matrix4x4Tools.c 
            ../../../../tools/AmMatrix/matrixMultiplicationOptimization.c
            ../../../../tools/AmMatrix/quaternions.c 
            #../../../../tools/AmMatrix/matrixProject.c
            #../../../../tools/AmMatrix/matrixProject.c
            ../../../../tools/AmMatrix/matrixCalculations.c 
            ../../../../tools/AmMatrix/matrixOpenGL.c 
            ../../../../tools/AmMatrix/solveLinearSystemGJ.c 
            ../../../../tools/AmMatrix/simpleRenderer.c 
            #ImageOperations dependencies
            ../../../../tools/ImageOperations/depthClassifier.c
            ../../../../tools/ImageOperations/convolutionFilter.c
            ../../../../tools/ImageOperations/imageFilters.c
            ../../../../tools/ImageOperations/findSubImage.c
            ../../../../tools/ImageOperations/imageOps.c   
            ../../../../tools/ImageOperations/patchComparison.c
            ../../../../tools/ImageOperations/patchComparison.h 
            ../../../../tools/ImageOperations/resize.c 
            #Rest of the stuff  
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
            TrajectoryParser/hashmap.c 
            TrajectoryParser/hashmap.h
            Interfaces/webInterface.c 
            Interfaces/webInterface.h
           ) 
target_link_libraries(OGLRendererSandbox m GL GLU X11 Codecs pthread ${PNG_Libs} ${JPG_Libs} ${GLEW_Libs} ${HTTP_Libs} ) 
 
 

set_target_properties(OGLRendererSandbox PROPERTIES 
                       #DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
                      )
