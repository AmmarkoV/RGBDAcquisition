project( Acquistion ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
 
   
    set(ACQUISITION_INGREDIENTS
                 Acquisition.c
                 Acquisition.h
                 pluginLinker.c
                 pluginLinker.h
                 acquisition_setup.h
              )
  
    add_library(Acquistion SHARED ${ACQUISITION_INGREDIENTS} ) 
    target_link_libraries(Acquistion rt dl m Timers CalibrationLibrary OperatingSystem ) 
 
 

set_target_properties(Acquistion PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
