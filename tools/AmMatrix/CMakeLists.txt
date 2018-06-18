project( AmMatrix )
cmake_minimum_required( VERSION 2.8.7 )
 
add_library(
            AmMatrix STATIC
            collisions.c
            quaternions.c
            matrix3x3Tools.c
            matrix4x4Tools.c
            matrixCalculations.c
            matrixCalculations.h
            matrixOpenGL.c
            matrixOpenGL.h 
            ocvStaging.c
            ocvStaging.h
            solveHomography.c
            solveLinearSystemGJ.c 
           )
target_link_libraries(AmMatrix m rt )


set_target_properties(AmMatrix PROPERTIES
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
