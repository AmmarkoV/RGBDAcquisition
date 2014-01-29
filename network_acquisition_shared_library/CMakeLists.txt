project( NetworkAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
  
add_library(NetworkAcquisition SHARED 
            main.c
            NetworkAcquisition.h
            daemon.c
            daemon.h
            networkFramework.c
            networkFramework.h 
            ../tools/Codecs/codecs.c
            ../tools/Codecs/codecs.h
            ../tools/Codecs/jpgInput.c
            ../tools/Codecs/jpgInput.h
           ) 
target_link_libraries(NetworkAcquisition rt m CalibrationLibrary OperatingSystem) 
 
 

set_target_properties(NetworkAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
