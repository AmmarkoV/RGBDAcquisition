project( TemplateAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
   

IF( ENABLE_JPG ) 
 MESSAGE("Enabling JPG support on Template Acquisition..")
 set(JPG_Libs jpeg ) 
 add_definitions(-DENABLE_JPG)
 add_definitions(-DUSE_CODEC_LIBRARY)
ENDIF( ENABLE_JPG )


IF( ENABLE_PNG ) 
 MESSAGE("Enabling PNG support on Template Acquisition..")
 set(PNG_Libs png ) 
 add_definitions(-DENABLE_PNG)
 add_definitions(-DUSE_CODEC_LIBRARY)
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
