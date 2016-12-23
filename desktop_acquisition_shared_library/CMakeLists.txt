project( DesktopAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
    
add_library(DesktopAcquisition SHARED 
            DesktopAcquisition.c
            DesktopAcquisition.h
           ) 
target_link_libraries(DesktopAcquisition rt m CalibrationLibrary OperatingSystem Codecs XwdLib X11 ) 
 
 

set_target_properties(DesktopAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
