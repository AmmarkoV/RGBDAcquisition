project( DisparityMapping ) 
cmake_minimum_required( VERSION 2.8.7 )   
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
  

add_library(DisparityMapping  SHARED  DisparityMapping.h DisparityMapping.cpp stereo_calibrate.cpp stereo_calibrate.h sgbm.cpp sgbm.h opticalFlow.cpp opticalFlow.h imageStream.cpp imageStream.h stereo_calib.cpp stereo_calib.h ) 
target_link_libraries(DisparityMapping rt dl m Acquisition OperatingSystem Timers CalibrationLibrary ${OpenCV_LIBRARIES} ${LocationService_Libs} )
set_target_properties(DisparityMapping PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(DisparityMapping Acquisition OperatingSystem Timers CalibrationLibrary)  
       


set_target_properties(DisparityMapping PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
