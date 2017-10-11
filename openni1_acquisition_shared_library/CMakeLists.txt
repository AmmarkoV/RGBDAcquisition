project( OpenNI1Acquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  

set(OpenNI1_DIR "${CMAKE_SOURCE_DIR}/3dparty")
set(OpenNI1_Includes ${CMAKE_SOURCE_DIR}/3dparty/OpenNI/Include/ )
  
FIND_PACKAGE(OpenNI1 REQUIRED)
INCLUDE_DIRECTORIES(${OPENNI1_INCLUDE_DIR} ${OpenNI1_Includes})
LINK_DIRECTORIES(${OPENNI1_LIBRARY_DIRS})
  


add_library(OpenNI1Acquisition SHARED OpenNI1Acquisition.cpp OpenNI1Acquisition.h ) 
target_link_libraries(OpenNI1Acquisition rt CalibrationLibrary ${OPENNI1_LIBRARIES}  OpenNI ) 


set_target_properties(OpenNI1Acquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
