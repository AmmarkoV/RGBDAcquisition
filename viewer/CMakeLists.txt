project( Viewer ) 
cmake_minimum_required( VERSION 2.8.7 )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Viewer main.c ) 
#${OpenCV_LIBS} ?
target_link_libraries(Viewer rt dl m Acquisition OperatingSystem Timers CalibrationLibrary ${OpenCV_LIBRARIES} ${LocationService_Libs} )
set_target_properties(Viewer PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(Viewer Acquisition OperatingSystem Timers CalibrationLibrary)  
       
 
set_target_properties(Viewer PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )



add_executable(ViewerNew main.cpp ) 
#${OpenCV_LIBS} ?
target_link_libraries(ViewerNew rt dl m Acquisition OperatingSystem Timers CalibrationLibrary ${OpenCV_LIBRARIES} ${LocationService_Libs} )
set_target_properties(ViewerNew PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(ViewerNew Acquisition OperatingSystem Timers CalibrationLibrary) 
set_target_properties(ViewerNew PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
