project( Processors ) 
cmake_minimum_required( VERSION 2.8.7 ) 

add_subdirectory (ViewpointChange/)
add_subdirectory (BlobDetector/)
 

IF( ENABLE_OPENCV ) 
IF(ENABLE_OPENCV_BASED_TOOLS) 
add_subdirectory (ObstacleDetector/)
add_subdirectory (DisparityMapping/)
ELSE (ENABLE_OPENCV_BASED_TOOLS)
 MESSAGE("DisparityMapping will not be compiled , OpenCV stuff missing? ( check ENABLE_OPENCV_BASED_TOOLS flag )..")
ENDIF(ENABLE_OPENCV_BASED_TOOLS)
ENDIF( ENABLE_OPENCV )

