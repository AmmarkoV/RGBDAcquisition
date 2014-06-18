project( TemplateAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
   

IF( ENABLE_JPG ) 
 set(JPG_Libs jpg ) 
ENDIF( ENABLE_JPG )


IF( ENABLE_PNG ) 
 set(PNG_Libs png ) 
ENDIF( ENABLE_PNG )



add_library(TemplateAcquisition SHARED 
            TemplateAcquisition.c
            TemplateAcquisition.h
            templateAcquisitionHelper.c
            templateAcquisitionHelper.h 
           ) 
target_link_libraries(TemplateAcquisition rt m CalibrationLibrary OperatingSystem Codecs  ${PNG_Libs} ${JPG_Libs} ) 
 
 

set_target_properties(TemplateAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
