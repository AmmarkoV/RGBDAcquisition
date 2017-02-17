project( TestBody ) 
cmake_minimum_required( VERSION 2.8.7 ) 
     
add_executable(TestBody main.c ) 
target_link_libraries(TestBody rt dl m Acquisition OperatingSystem Timers CalibrationLibrary ${LocationService_Libs} )
set_target_properties(TestBody PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(TestBody Acquisition OperatingSystem Timers CalibrationLibrary)  
       
 
set_target_properties(TestBody PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
