project( OpenGLAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
  

IF( ENABLE_JPG )
 MESSAGE("JPGs will be included in this OpenGL Codec build")
 set(JPG_Libs jpeg ) 
 set(JPG_Parts jpgInput.c jpgInput.h jpgExifexternal.c jpgExifexternal.h jpgExiforient_embed.c jpgExiforient_embed.h )
 set(JPG_Includes ${CMAKE_SOURCE_DIR}/3dparty/OpenNI2/Include/ )
 add_definitions(-DUSE_JPG_FILES)
 add_definitions(-DENABLE_JPG)
ENDIF( ENABLE_JPG )


IF( ENABLE_PNG )
 MESSAGE("PNGs will be included in this OpenGL Codec build")
 set(PNG_Libs png ) 
 set(PNG_Parts pngInput.c pngInput.h)
 set(PNG_Includes ${CMAKE_SOURCE_DIR}/3dparty/OpenNI2/Include/ )
 add_definitions(-DUSE_PNG_FILES)
 add_definitions(-DENABLE_PNG)
ENDIF( ENABLE_PNG )

 

add_library(OpenGLAcquisition SHARED 
            OpenGLAcquisition.c
            OpenGLAcquisition.h
            opengl_depth_and_color_renderer/src/Library/System/glx.h
            opengl_depth_and_color_renderer/src/Library/System/glx.c
            opengl_depth_and_color_renderer/src/Library/System/glx2.h
            opengl_depth_and_color_renderer/src/Library/System/glx2.c
            opengl_depth_and_color_renderer/src/Library/System/glx3.h
            opengl_depth_and_color_renderer/src/Library/System/glx3.c
            opengl_depth_and_color_renderer/src/Library/main.c
            opengl_depth_and_color_renderer/src/Library/OGLRendererSandbox.h
            #3D Models and how to load them  
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader.c
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader.h
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader_obj.c
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader_obj.h
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader_tri.c
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader_tri.h
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_processor.c
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_processor.h 
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader_hardcoded.c
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader_hardcoded.h 
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader_transform_joints.c
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_loader_transform_joints.h
            opengl_depth_and_color_renderer/src/Library/ModelLoader/tri_bvh_controller.h
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_converter.c
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_converter.h
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_editor.c
            opengl_depth_and_color_renderer/src/Library/ModelLoader/model_editor.h
            #Inverse Kinematics
            opengl_depth_and_color_renderer/src/Library/MotionCaptureLoader/ik/bvh_inverseKinematics.c
            opengl_depth_and_color_renderer/src/Library/MotionCaptureLoader/ik/hardcodedProblems_inverseKinematics.c
            #opengl_depth_and_color_renderer/src/Library/MotionCaptureLoader/ik/levmar.c 
            #Textures and how to load them 
            opengl_depth_and_color_renderer/src/Library/TextureLoader/texture_loader.c
            opengl_depth_and_color_renderer/src/Library/TextureLoader/texture_loader.h
            opengl_depth_and_color_renderer/src/Library/TextureLoader/image_proc.c
            opengl_depth_and_color_renderer/src/Library/TextureLoader/image_proc.h
            #OpenGL Rendering stuff  
            opengl_depth_and_color_renderer/src/Library/Rendering/FixedPipeline/ogl_fixed_pipeline_renderer.c
            opengl_depth_and_color_renderer/src/Library/Rendering/FixedPipeline/ogl_fixed_pipeline_renderer.h
            opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/render_buffer.c
            opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/render_buffer.h
            opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/uploadGeometry.h
            opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/uploadGeometry.c 
            opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/ogl_shader_pipeline_renderer.c
            opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/ogl_shader_pipeline_renderer.h
            opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/shader_loader.c
            opengl_depth_and_color_renderer/src/Library/Rendering/ShaderPipeline/shader_loader.h
            opengl_depth_and_color_renderer/src/Library/Rendering/ogl_rendering.c
            opengl_depth_and_color_renderer/src/Library/Rendering/ogl_rendering.h
            opengl_depth_and_color_renderer/src/Library/Rendering/tiledRenderer.h
            opengl_depth_and_color_renderer/src/Library/Rendering/tiledRenderer.c
            opengl_depth_and_color_renderer/src/Library/Rendering/downloadFromRenderer.c
            opengl_depth_and_color_renderer/src/Library/Rendering/downloadFromRenderer.h
            #Tools
            opengl_depth_and_color_renderer/src/Library/Tools/tools.h
            opengl_depth_and_color_renderer/src/Library/Tools/tools.c
            opengl_depth_and_color_renderer/src/Library/Tools/save_to_file.h 
            opengl_depth_and_color_renderer/src/Library/Tools/save_to_file.c
            opengl_depth_and_color_renderer/src/Library/Scene/scene.h
            opengl_depth_and_color_renderer/src/Library/Scene/scene.c
            opengl_depth_and_color_renderer/src/Library/Scene/control.h
            opengl_depth_and_color_renderer/src/Library/Scene/control.c
            opengl_depth_and_color_renderer/src/Library/Scene/photoShootingScene.h
            opengl_depth_and_color_renderer/src/Library/Scene/photoShootingScene.c
            #AmMatrix dependencies 
            ../tools/AmMatrix/quaternions.c
            ../tools/AmMatrix/quaternions.h
            ../tools/AmMatrix/matrix3x3Tools.c
            ../tools/AmMatrix/matrix3x3Tools.h
            ../tools/AmMatrix/matrix4x4Tools.c
            ../tools/AmMatrix/matrix4x4Tools.h
            #../tools/AmMatrix/matrixProject.c
            #../tools/AmMatrix/matrixProject.h
            ../tools/AmMatrix/matrixCalculations.c
            ../tools/AmMatrix/matrixCalculations.h
            ../tools/AmMatrix/matrixOpenGL.c
            ../tools/AmMatrix/matrixOpenGL.h
            ../tools/AmMatrix/solveLinearSystemGJ.c
            ../tools/AmMatrix/solveLinearSystemGJ.h 
            #ImageOperations dependencies
            ../tools/ImageOperations/depthClassifier.c
            ../tools/ImageOperations/convolutionFilter.c
            ../tools/ImageOperations/imageFilters.c
            ../tools/ImageOperations/findSubImage.c
            ../tools/ImageOperations/imageOps.c   
            ../tools/ImageOperations/patchComparison.c
            ../tools/ImageOperations/patchComparison.h 
            ../tools/ImageOperations/resize.c 
            #Rest of the stuff  
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/InputParser_C.c
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/InputParser_C.h
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/TrajectoryParser.c
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/TrajectoryParser.h 
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/TrajectoryParserDataStructures.c
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/TrajectoryParserDataStructures.h 
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/TrajectoryCalculator.c
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/TrajectoryCalculator.h  
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/TrajectoryPrimitives.c 
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/TrajectoryPrimitives.h
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/hashmap.c 
            opengl_depth_and_color_renderer/src/Library/TrajectoryParser/hashmap.h
            opengl_depth_and_color_renderer/src/Library/Interfaces/webInterface.c 
            opengl_depth_and_color_renderer/src/Library/Interfaces/webInterface.h
           ) 
target_link_libraries(OpenGLAcquisition rt m CalibrationLibrary GL GLU X11 Codecs pthread ${PNG_Libs} ${JPG_Libs}) 
 
 

set_target_properties(OpenGLAcquisition PROPERTIES 
                       #DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
