project( OpenNI2Acquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  

set(OpenNI2_DIR "${CMAKE_SOURCE_DIR}/3dparty" )
   
FIND_PACKAGE(OpenNI2 REQUIRED)
INCLUDE_DIRECTORIES(${OPENNI2_INCLUDE_DIR})
LINK_DIRECTORIES(${OPENNI2_LIBRARY_DIRS})


add_library(OpenNI2Acquisition SHARED OpenNI2Acquisition.cpp OpenNI2Acquisition.h ${OPENNI2_INCLUDES} ) 
target_link_libraries(OpenNI2Acquisition rt CalibrationLibrary ${OPENNI2_LIBRARIES} ) 


set_target_properties(OpenNI2Acquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
