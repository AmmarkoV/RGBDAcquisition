project( GrabberMux ) 
cmake_minimum_required( VERSION 2.8.7 ) 
    
add_executable(GrabberMux main.c ) 
target_link_libraries(GrabberMux rt dl m RGBDAcquisition AcquisitionMux OperatingSystem Timers CalibrationLibrary ${LocationService_Libs} )
set_target_properties(GrabberMux PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(GrabberMux Acquisition AcquisitionMux OperatingSystem Timers CalibrationLibrary)  
       
 
set_target_properties(GrabberMux PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
