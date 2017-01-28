project( FreenectAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  

set(Freenect_DIR "${CMAKE_SOURCE_DIR}/3dparty/" ) 

FIND_PACKAGE(Freenect REQUIRED)
INCLUDE_DIRECTORIES(${FREENECT_INCLUDE_DIR})
LINK_DIRECTORIES(${FREENECT_LIBRARY_DIRS})


add_library(FreenectAcquisition SHARED FreenectAcquisition.c FreenectAcquisition.h ${FREENECT_INCLUDES} ) 
target_link_libraries(FreenectAcquisition rt dl m Timers CalibrationLibrary OperatingSystem ${FREENECT_LIBRARIES} freenect_sync )  
 

set_target_properties(FreenectAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
