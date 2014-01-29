project( OpenNI1Acquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/3dparty ${CMAKE_MODULE_PATH})
  
FIND_PACKAGE(findOpenNI1 REQUIRED)
INCLUDE_DIRECTORIES(${OPENNI1_INCLUDE_DIR})
LINK_DIRECTORIES(${OPENNI1_LIBRARY_DIRS})
 
 
add_library(OpenNI1Acquisition SHARED OpenNI1Acquisition.cpp OpenNI1Acquisition.h  ${OPENNI1_INCLUDES}) 
target_link_libraries(OpenNI1Acquisition rt CalibrationLibrary ${OPENNI1_LIBRARIES} ) 


set_target_properties(OpenNI1Acquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
