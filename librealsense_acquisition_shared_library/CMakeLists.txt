project( RealsenseAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  

set(Realsense_DIR "${CMAKE_SOURCE_DIR}/3dparty" )

FIND_PACKAGE(librealsense REQUIRED)
INCLUDE_DIRECTORIES(${FREENECT_INCLUDE_DIR})
LINK_DIRECTORIES(${FREENECT_LIBRARY_DIRS})


add_library(RealsenseAcquisition SHARED RealsenseAcquisition.c RealsenseAcquisition.h ${FREENECT_INCLUDES} ) 
target_link_libraries(RealsenseAcquisition rt dl m Timers CalibrationLibrary OperatingSystem ${FREENECT_LIBRARIES} )  
 

set_target_properties(RealsenseAcquisition PROPERTIES 
                       #DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
