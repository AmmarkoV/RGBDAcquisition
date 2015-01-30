project( ImageOperations ) 
cmake_minimum_required( VERSION 2.8.7 )
 
add_library(
            ImageOperations STATIC 
            depthClassifier.c
            convolutionFilter.c
            imageFilters.c
            findSubImage.c
            imageOps.c 
            joinMPO.c
            main.c
            ocr.c
            ocr.h 
            patchComparison.c
            patchComparison.h
            patternSets.c
            resize.c
            splitMPO.c 
           )

target_link_libraries(ImageOperations rt dl m  )
set_target_properties(ImageOperations PROPERTIES DEBUG_POSTFIX "D")   
         
 

set_target_properties(ImageOperations PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
 
