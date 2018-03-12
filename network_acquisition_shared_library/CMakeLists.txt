project( NetworkAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )   
  


set(AmmarServer_Parts daemon.c )
IF(ENABLE_AMMARSERVER_BROADCAST)
 set(AmmarServer_Parts daemon_ammarserver.c )
 set(AmmarServer_Libs AmmarServer )
 add_definitions(-DUSE_AMMARSERVER)
ENDIF(ENABLE_AMMARSERVER_BROADCAST)

add_library(NetworkAcquisition SHARED 
            main.c
            NetworkAcquisition.h
            ${AmmarServer_Parts}
            daemon.h
            networkFramework.c
            networkFramework.h 
            ../tools/Codecs/codecs.c
            ../tools/Codecs/codecs.h
            ../tools/Codecs/jpgInput.c
            ../tools/Codecs/jpgInput.h
           ) 
target_link_libraries(NetworkAcquisition rt m CalibrationLibrary OperatingSystem ${AmmarServer_Libs} ) 
 
 

set_target_properties(NetworkAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
