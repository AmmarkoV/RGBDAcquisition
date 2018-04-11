project( GrabberSegment ) 
cmake_minimum_required( VERSION 2.8.7 ) 
    
add_executable(GrabberSegment main.c ) 
target_link_libraries(GrabberSegment rt dl m RGBDAcquisition AcquisitionSegment OperatingSystem Timers CalibrationLibrary ${LocationService_Libs} )
set_target_properties(GrabberSegment PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(GrabberSegment Acquisition AcquisitionSegment OperatingSystem Timers CalibrationLibrary)  
       
 
set_target_properties(GrabberSegment PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
