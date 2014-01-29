project( AcquisitionSegment ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
 
add_library(AcquisitionSegment SHARED 
            AcquisitionSegment.c
            AcquisitionSegment.h
            colorSelector.c
            colorSelector.h
            combineRGBAndDepthOutput.c
            combineRGBAndDepthOutput.h 
            depthSelector.c
            depthSelector.h
            imageProcessing.c
            imageProcessing.h
            ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrix4x4Tools.c
            ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrix4x4Tools.h
            ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrix3x3Tools.c
            ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrix3x3Tools.h 
            ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrixCalculations.c
            ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrixCalculations.h
            ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/solveLinearSystemGJ.c
            ../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/solveLinearSystemGJ.h
            ) 
target_link_libraries(AcquisitionSegment m CalibrationLibrary ) 
 
 

set_target_properties(AcquisitionSegment PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
