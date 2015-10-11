project( Grabber ) 
cmake_minimum_required( VERSION 2.8.7 ) 
     
add_executable(Grabber main.c ) 
target_link_libraries(Grabber rt dl m Acquisition OperatingSystem Timers CalibrationLibrary ${LocationService_Libs} )
set_target_properties(Grabber PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(Grabber Acquisition OperatingSystem Timers CalibrationLibrary)  
       
 
set_target_properties(Grabber PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
