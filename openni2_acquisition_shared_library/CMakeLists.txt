project( OpenNI2Acquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  

set(OpenNI2_DIR "${CMAKE_SOURCE_DIR}/3dparty" )
   
FIND_PACKAGE(OpenNI2 REQUIRED)

set(OpenNI2_Parts OpenNI2Acquisition.cpp  OpenNI2Acquisition.h )
set(OpenNI2_Includes ${CMAKE_SOURCE_DIR}/3dparty/OpenNI2/Include/ )

IF( ENABLE_NITE2 )
 set(OpenNI2_Parts OpenNI2Acquisition.cpp OpenNI2Acquisition.h Nite2.cpp Nite2.h )
 set(OpenNI2_Includes ${CMAKE_SOURCE_DIR}/3dparty/OpenNI2/Include/ ${CMAKE_SOURCE_DIR}/3dparty/NiTE-Linux-x64-2.2/Include )
 set(OpenNI2_Libraries ${CMAKE_SOURCE_DIR}/3dparty/NiTE-Linux-x64-2.2/Redist/libNiTE2.so)
 add_definitions(-DMOD_NITE2)
ENDIF( ENABLE_NITE2 )


IF( ENABLE_FACEDETECTION_DEV )
 add_definitions(-DMOD_FACEDETECTION)
 find_package(OpenCV REQUIRED)
 INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS}) 
 set(FaceDetection_Parts FaceDetection.cpp FaceDetection.h ../tools/Calibration/calibration.c  ../tools/Calibration/calibration.h ../tools/ImageOperations/imageOps.c  ../tools/ImageOperations/imageOps.h )
 set(FaceDetection_Libraries ${OpenCV_LIBRARIES} )
ENDIF( ENABLE_FACEDETECTION_DEV )


INCLUDE_DIRECTORIES(${OPENNI2_INCLUDE_DIR} ${OpenNI2_Includes} )
LINK_DIRECTORIES(${OPENNI2_LIBRARY_DIRS} )


add_library(OpenNI2Acquisition SHARED ${OpenNI2_Parts} ${FaceDetection_Parts} ) 
target_link_libraries(OpenNI2Acquisition rt CalibrationLibrary ${OPENNI2_LIBRARIES} ${OpenNI2_Libraries} ${FaceDetection_Libraries} ) 


set_target_properties(OpenNI2Acquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
