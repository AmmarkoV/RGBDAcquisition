project( AcquisitionSegment ) 
cmake_minimum_required( VERSION 2.8.7 )   
 
add_library(AcquisitionSegment STATIC 
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
            automaticPlaneSegmentation.c
            automaticPlaneSegmentation.h
            ../tools/Quasirandomness/quasirandomness.c
            ../tools/Quasirandomness/quasirandomness.h
            ../tools/AmMatrix/matrix4x4Tools.c
            ../tools/AmMatrix/matrix4x4Tools.h
            ../tools/AmMatrix/matrix3x3Tools.c
            ../tools/AmMatrix/matrix3x3Tools.h 
            ../tools/AmMatrix/matrixCalculations.c
            ../tools/AmMatrix/matrixCalculations.h
            ../tools/AmMatrix/solveLinearSystemGJ.c
            ../tools/AmMatrix/solveLinearSystemGJ.h
            ) 
target_link_libraries(AcquisitionSegment m CalibrationLibrary ) 
 
 

set_target_properties(AcquisitionSegment PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
