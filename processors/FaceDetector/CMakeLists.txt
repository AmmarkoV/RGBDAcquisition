project( FaceDetector ) 
cmake_minimum_required( VERSION 2.8.7 )   
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
  
IF( ENABLE_AMMARSERVER )
 MESSAGE("Shaders will be included in this codec build")
 set(HTTP_Libs AmmClient )   
 set(HTTP_Includes "../../../../AmmarServer/src/AmmClient/AmmClient.h" )  
ENDIF( ENABLE_AMMARSERVER )

add_library(FaceDetector  SHARED  FaceDetector.h FaceDetector.cpp ${HTTP_Includes} ) 
target_link_libraries(FaceDetector rt dl m Acquisition OperatingSystem Timers CalibrationLibrary ${OpenCV_LIBRARIES} ${LocationService_Libs} ${HTTP_Libs} )
set_target_properties(FaceDetector PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(FaceDetector Acquisition OperatingSystem Timers CalibrationLibrary)  
       


set_target_properties(FaceDetector PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
