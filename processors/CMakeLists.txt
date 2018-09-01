project( Processors ) 
cmake_minimum_required( VERSION 2.8.7 ) 

add_subdirectory (ViewpointChange/)
add_subdirectory (BlobDetector/)
 

IF( ENABLE_OPENCV ) 
IF(ENABLE_OPENCV_BASED_TOOLS) 
add_subdirectory (ObstacleDetector/)
#add_subdirectory (DisparityMapping/)
add_subdirectory (FaceDetector/)
ELSE (ENABLE_OPENCV_BASED_TOOLS)
 MESSAGE("DisparityMapping will not be compiled , OpenCV stuff missing? ( check ENABLE_OPENCV_BASED_TOOLS flag )..")
ENDIF(ENABLE_OPENCV_BASED_TOOLS)
ENDIF( ENABLE_OPENCV )



#-------------------------------------------------------
  IF(ENABLE_PROCESSOR_BODYTRACKER) 
    MESSAGE("Will use BodyTracker")
    add_subdirectory (BodyTracker/) 
    add_subdirectory (BodyTrackerExe/) 
  ELSE (ENABLE_PROCESSOR_BODYTRACKER)
    MESSAGE("Will not use BodyTracker")
  ENDIF(ENABLE_PROCESSOR_BODYTRACKER)

#-------------------------------------------------------
  IF(ENABLE_PROCESSOR_DARKNET) 
    MESSAGE("Will use DarknetProcessor")
    add_subdirectory (DarknetProcessor/) 
  ELSE (ENABLE_PROCESSOR_DARKNET)
    MESSAGE("Will not use DarknetProcessor")
  ENDIF(ENABLE_PROCESSOR_DARKNET)
#-------------------------------------------------------
  IF(ENABLE_PROCESSOR_MOVIDIUS) 
    MESSAGE("Will use Movidius")
    add_subdirectory (Movidius/) 
  ELSE (ENABLE_PROCESSOR_MOVIDIUS)
    MESSAGE("Will not use Movidius")
  ENDIF(ENABLE_PROCESSOR_MOVIDIUS)
#-------------------------------------------------------
