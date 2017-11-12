project( ImageOperations ) 
cmake_minimum_required( VERSION 2.8.7 )
 



IF( ENABLE_JPG )
 MESSAGE("JPGs will be included in this codec build")
 set(JPG_Libs jpeg ) 
ENDIF( ENABLE_JPG )


IF( ENABLE_PNG )
 MESSAGE("PNGs will be included in this codec build")
 set(PNG_Libs png ) 
ENDIF( ENABLE_PNG )



add_library(
            ImageOperations STATIC 
            depthClassifier.c
            convolutionFilter.c
            imageFilters.c
            findSubImage.c
            imageOps.c 
            joinMPO.c 
            ocr.c
            ocr.h 
            patchComparison.c
            patchComparison.h
            patternSets.c
            projection.c
            resize.c
            splitMPO.c 
            bilateralFilter.c
            bilateralFilter.h
            medianFilter.c
            medianFilter.h
            summedAreaTables.c
            summedAreaTables.h
            compareQuality.c
            compareQuality.h
            learnImage.c
            learnImage.h
 
            special/dericheRecursiveGaussian.h
            special/dericheRecursiveGaussian.c
            special/constantTimeBilateralFiltering.h
            special/constantTimeBilateralFiltering.c

            tools/imageMatrix.h
            tools/imageMatrix.c
           )

target_link_libraries(ImageOperations rt dl m  )
set_target_properties(ImageOperations PROPERTIES DEBUG_POSTFIX "D")   
         
add_executable(imageopsutility imageOperationsTester.c ) 
target_link_libraries(imageopsutility rt dl m Codecs ImageOperations Timers ${PNG_Libs} ${JPG_Libs} ) 
add_dependencies(imageopsutility Codecs ImageOperations)  
 

set_target_properties(ImageOperations PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )


set_target_properties(imageopsutility PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )

 
