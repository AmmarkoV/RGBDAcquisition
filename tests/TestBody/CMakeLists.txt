project( TestBody ) 
cmake_minimum_required( VERSION 2.8.7 ) 
     
add_executable(TestBody main.c  smartbody_tri_converter.c
../../../tools/AmMatrix/matrixProject.c
../../../tools/AmMatrix/matrixProject.h
../../../tools/Primitives/jsonCocoSkeleton.h
../../../tools/Primitives/jsonCocoSkeleton.c
../../../tools/Primitives/skeleton.h
../../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_loader_transform_joints.c
../../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_loader_transform_joints.h
../../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_loader_tri.c
../../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_loader_tri.h
../../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_processor.c
../../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_processor.h
../../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/TrajectoryParser/InputParser_C.c
../../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/TrajectoryParser/InputParser_C.h
) 
target_link_libraries(TestBody rt dl m Acquisition OperatingSystem Timers CalibrationLibrary ${LocationService_Libs} )
set_target_properties(TestBody PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(TestBody Acquisition OperatingSystem Timers CalibrationLibrary)  
       
 
set_target_properties(TestBody PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
