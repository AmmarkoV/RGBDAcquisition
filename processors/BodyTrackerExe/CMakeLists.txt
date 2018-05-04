project( BodyTracker ) 
cmake_minimum_required( VERSION 2.8.7 )   

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
 
add_executable(BodyTracker main.cpp ) 
target_link_libraries(BodyTracker rt dl m RGBDAcquisition OperatingSystem Timers CalibrationLibrary ${OpenCV_LIBRARIES} ${LocationService_Libs} ) 
 
add_dependencies(BodyTracker Acquisition OperatingSystem Timers CalibrationLibrary) 

set_target_properties(BodyTracker PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
 
