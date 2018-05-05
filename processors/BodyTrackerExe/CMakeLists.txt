project( BodyTrackerExe ) 
cmake_minimum_required( VERSION 2.8.7 )   

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
 
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
ADD_LIBRARY(FORTHUpperBodyGestureTrackerAdapter SHARED IMPORTED)
SET_TARGET_PROPERTIES(FORTHUpperBodyGestureTrackerAdapter PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libFORTHUpperBodyGestureTrackerAdapter.so) 


ADD_LIBRARY(ArticulatedTracking SHARED IMPORTED)
SET_TARGET_PROPERTIES(ArticulatedTracking PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libArticulatedTracking.so) 

ADD_LIBRARY(MBVAcquisition SHARED IMPORTED)
SET_TARGET_PROPERTIES(MBVAcquisition PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libAcquisition.so) 

ADD_LIBRARY(MBVCore SHARED IMPORTED)
SET_TARGET_PROPERTIES(MBVCore PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libCore.so )

ADD_LIBRARY(cudpp SHARED IMPORTED)
SET_TARGET_PROPERTIES(cudpp PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libcudpp.so ) 

ADD_LIBRARY(MBVDecoding SHARED IMPORTED)
SET_TARGET_PROPERTIES(MBVDecoding PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libDecoding.so)
  
ADD_LIBRARY(HandTracker SHARED IMPORTED)
SET_TARGET_PROPERTIES(HandTracker PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libHandTrackerLib.so)

ADD_LIBRARY(MBVLibraries SHARED IMPORTED)
SET_TARGET_PROPERTIES(MBVLibraries PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libLibraries.so)
 
ADD_LIBRARY(MBVOpenMesh SHARED IMPORTED)
SET_TARGET_PROPERTIES(MBVOpenMesh PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libMBVOpenMesh.so)

ADD_LIBRARY(MBVOptimization SHARED IMPORTED)
SET_TARGET_PROPERTIES(MBVOptimization PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libOptimization.so)

ADD_LIBRARY(MBVParticleFilter SHARED IMPORTED)
SET_TARGET_PROPERTIES(MBVParticleFilter PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libParticleFilter.so)
 
ADD_LIBRARY(MBVPhysics SHARED IMPORTED)
SET_TARGET_PROPERTIES(MBVPhysics PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libPhysics.so)

ADD_LIBRARY(MBVRendering SHARED IMPORTED)
SET_TARGET_PROPERTIES(MBVRendering PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libRendering.so)

#BOOST ----------------------------------------------------------------------------------------------------------------------------------------------------------
ADD_LIBRARY(BOOSTCHRONO SHARED IMPORTED)
SET_TARGET_PROPERTIES(BOOSTCHRONO PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libboost_chrono.so.1.57.0)
 
ADD_LIBRARY(BOOSTCONTEXT SHARED IMPORTED)
SET_TARGET_PROPERTIES(BOOSTCONTEXT PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libboost_context.so.1.57.0)

ADD_LIBRARY(BOOSTCOROUTINE SHARED IMPORTED)
SET_TARGET_PROPERTIES(BOOSTCOROUTINE PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libboost_coroutine.so.1.57.0)
 
ADD_LIBRARY(BOOSTFILESYSTEM SHARED IMPORTED)
SET_TARGET_PROPERTIES(BOOSTFILESYSTEM PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libboost_filesystem.so.1.57.0)

ADD_LIBRARY(BOOSTLOG SHARED IMPORTED)
SET_TARGET_PROPERTIES(BOOSTLOG PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libboost_log.so.1.57.0)

ADD_LIBRARY(BOOSTPROGRAMOPTIONS SHARED IMPORTED)
SET_TARGET_PROPERTIES(BOOSTPROGRAMOPTIONS PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libboost_program_options.so.1.57.0)

ADD_LIBRARY(BOOSTSERIALIZATION SHARED IMPORTED)
SET_TARGET_PROPERTIES(BOOSTSERIALIZATION PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libboost_serialization.so.1.57.0)

ADD_LIBRARY(BOOSTSYSTEM SHARED IMPORTED)
SET_TARGET_PROPERTIES(BOOSTSYSTEM PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libboost_system.so.1.57.0)

ADD_LIBRARY(BOOSTTHREAD SHARED IMPORTED)
SET_TARGET_PROPERTIES(BOOSTTHREAD PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libboost_thread.so.1.57.0)
 
#OTHER LIBRARIES ----------------------------------------------------------------------------------------------------------------------------------------------------------

ADD_LIBRARY(GLEW SHARED IMPORTED)
SET_TARGET_PROPERTIES(GLEW PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libGLEW.so.1.13)
			                                 
ADD_LIBRARY(OpenMesh SHARED IMPORTED)
SET_TARGET_PROPERTIES(OpenMesh PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libOpenMeshCore.so.3.3)

ADD_LIBRARY(BulletCollision SHARED IMPORTED)
SET_TARGET_PROPERTIES(BulletCollision PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libBulletCollision.so.2.81 )

ADD_LIBRARY(BulletDynamics SHARED IMPORTED)
SET_TARGET_PROPERTIES(BulletDynamics PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libBulletDynamics.so.2.81 )

ADD_LIBRARY(BulletSoftBody SHARED IMPORTED)
SET_TARGET_PROPERTIES(BulletSoftBody PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libBulletSoftBody.so.2.81 )

ADD_LIBRARY(BulletLinearMath SHARED IMPORTED)
SET_TARGET_PROPERTIES(BulletLinearMath PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libLinearMath.so.2.81 )     


ADD_LIBRARY(OpenCVObjDetect SHARED IMPORTED)
SET_TARGET_PROPERTIES(OpenCVObjDetect PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libopencv_objdetect.so.3.2)     

ADD_LIBRARY(OpenCVVideo SHARED IMPORTED)
SET_TARGET_PROPERTIES(OpenCVVideo PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libopencv_video.so.3.2)  

ADD_LIBRARY(OpenCVVideoIO SHARED IMPORTED)
SET_TARGET_PROPERTIES(OpenCVVideoIO PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libopencv_videoio.so.3.2)  

ADD_LIBRARY(OpenCVCalib3D SHARED IMPORTED)
SET_TARGET_PROPERTIES(OpenCVCalib3D PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libopencv_calib3d.so.3.2)   

ADD_LIBRARY(OpenCVCore SHARED IMPORTED)
SET_TARGET_PROPERTIES(OpenCVCore PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libopencv_core.so.3.2)

ADD_LIBRARY(OpenCVHighGUI SHARED IMPORTED)
SET_TARGET_PROPERTIES(OpenCVHighGUI PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BodyTracker/forth_skeleton_tracker_redist/libopencv_highgui.so.3.2) 

#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------

add_executable(BodyTrackerExe main.cpp ) 
target_link_libraries(
                      BodyTrackerExe 
                      rt 
                      dl 
                      m 
                      RGBDAcquisition 
                      OperatingSystem 
                      Timers 
                      CalibrationLibrary
                      #FORTH libs here
                      FORTHUpperBodyGestureTrackerAdapter
                      ArticulatedTracking
                      #Ugly dependencies here
                      MBVAcquisition
                      MBVCore
                      cudpp
                      MBVDecoding
                      HandTracker
                      MBVLibraries
                      MBVOpenMesh 
                      MBVOptimization
                      MBVParticleFilter
                      MBVPhysics 
                      MBVRendering
                      #Cancer boost dependencies here
                      BOOSTCHRONO  
                      BOOSTCONTEXT
                      BOOSTCOROUTINE             
                      BOOSTFILESYSTEM 
                      BOOSTLOG             
                      BOOSTPROGRAMOPTIONS      
                      BOOSTSERIALIZATION 
                      BOOSTSYSTEM 
                      BOOSTTHREAD 
                      GLEW 
                      OpenMesh
                      BulletCollision 
                      BulletDynamics 
                      BulletSoftBody
                      #Use given OpenCV libraries to avoid problems
                      OpenCVObjDetect
                      OpenCVVideo  
                      OpenCVVideoIO
                      OpenCVCalib3D
                      OpenCVCore
                      #Use system OpenCV libraries ( this might create problems )
                      #${OpenCV_LIBRARIES} 
                      ${LocationService_Libs} 
                      ) 
 
add_dependencies(BodyTrackerExe RGBDAcquisition OperatingSystem Timers CalibrationLibrary) 

set_target_properties(BodyTrackerExe PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
 
