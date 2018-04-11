project( Viewer ) 
cmake_minimum_required( VERSION 2.8.7 )

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})


#Old code for viewer..
#add_executable(ViewerOld main.c ) 
#target_link_libraries(ViewerOld rt dl m Acquisition OperatingSystem Timers CalibrationLibrary ${OpenCV_LIBRARIES} ${LocationService_Libs} )
#set_target_properties(ViewerOld PROPERTIES DEBUG_POSTFIX "D")
#add_dependencies(ViewerOld Acquisition OperatingSystem Timers CalibrationLibrary)        
#set_target_properties(ViewerOld PROPERTIES 
#                       DEBUG_POSTFIX "D"
#                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#                      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
#                      )



add_executable(Viewer main.cpp )  
target_link_libraries(Viewer rt dl m RGBDAcquisition OperatingSystem Timers CalibrationLibrary ${OpenCV_LIBRARIES} ${LocationService_Libs} )
set_target_properties(Viewer PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(Viewer Acquisition OperatingSystem Timers CalibrationLibrary) 
set_target_properties(Viewer PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
