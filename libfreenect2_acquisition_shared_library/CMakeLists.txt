project( Freenect2Acquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  

set(Freenect2_DIR "${CMAKE_SOURCE_DIR}/3dparty" )

FIND_PACKAGE(Freenect2 REQUIRED)
INCLUDE_DIRECTORIES(${FREENECT2_INCLUDE_DIR})
LINK_DIRECTORIES(${FREENECT2_LIBRARY_DIRS})


add_library(Freenect2Acquisition SHARED Freenect2Acquisition.c Freenect2Acquisition.h ${FREENECT2_INCLUDES} ) 
target_link_libraries(FreenectAcquisition rt dl m Timers CalibrationLibrary OperatingSystem ${FREENECT2_LIBRARIES} )  
 

set_target_properties(Freenect2Acquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
