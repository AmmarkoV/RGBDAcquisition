project( SharedMemory ) 
cmake_minimum_required( VERSION 2.8.7 )   
 
 
add_library(SharedMemory SHARED SharedMemory.c ) 
target_link_libraries(SharedMemory rt dl m Timers CalibrationLibrary OperatingSystem Codecs   ) 
 
    set_target_properties(SharedMemory PROPERTIES 
                          DEBUG_POSTFIX "D"
                          ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                         )


 
