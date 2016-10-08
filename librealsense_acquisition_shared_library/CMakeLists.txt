project( RealsenseAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  

set(Realsense_DIR "${CMAKE_SOURCE_DIR}/3dparty/librealsense" )

#FIND_PACKAGE(librealsense REQUIRED)
INCLUDE_DIRECTORIES("${Realsense_DIR}/include")
#LINK_DIRECTORIES("${Realsense_DIR}/build/devel/lib/librealsense.so")


add_library(RealsenseAcquisition SHARED RealsenseAcquisition.c RealsenseAcquisition.h  ) 
target_link_libraries(RealsenseAcquisition rt dl m Timers CalibrationLibrary OperatingSystem "${Realsense_DIR}/build/devel/lib/librealsense.so"  )  
 

set_target_properties(RealsenseAcquisition PROPERTIES 
                       #DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
