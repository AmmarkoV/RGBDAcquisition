project( CalibrationLibrary ) 
cmake_minimum_required( VERSION 2.8.7 )   
 
add_library(CalibrationLibrary STATIC 
            calibration.c calibration.h 
            undistort.c   undistort.h  
            transform.c   transform.h 
            ../AmMatrix/quaternions.c
            ../AmMatrix/quaternions.h
            ../AmMatrix/matrix4x4Tools.c
            ../AmMatrix/matrix4x4Tools.h
            ../AmMatrix/matrix3x3Tools.c
            ../AmMatrix/matrix3x3Tools.h 
            ../AmMatrix/matrixCalculations.c
            ../AmMatrix/matrixCalculations.h
            ../AmMatrix/matrixOpenGL.c
            ../AmMatrix/matrixOpenGL.h
            ../AmMatrix/solveLinearSystemGJ.c
            ../AmMatrix/solveLinearSystemGJ.h
            ../AmMatrix/ocvStaging.c
            ../AmMatrix/ocvStaging.h
            ../AmMatrix/solveHomography.c
            ../AmMatrix/solveHomography.h
           ) 
target_link_libraries(CalibrationLibrary m ) 
 

set_target_properties(CalibrationLibrary PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
