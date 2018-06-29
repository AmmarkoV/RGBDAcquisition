project( FaceDetector ) 
cmake_minimum_required( VERSION 2.8.7 )   
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
  

add_library(FaceDetector  SHARED  FaceDetector.h FaceDetector.cpp  ) 
target_link_libraries(FaceDetector rt dl m Acquisition OperatingSystem Timers CalibrationLibrary ${OpenCV_LIBRARIES} ${LocationService_Libs} )
set_target_properties(FaceDetector PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(FaceDetector Acquisition OperatingSystem Timers CalibrationLibrary)  
       


set_target_properties(FaceDetector PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
