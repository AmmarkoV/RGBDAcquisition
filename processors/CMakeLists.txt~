project( Processors ) 
cmake_minimum_required( VERSION 2.8.7 ) 

add_subdirectory (ViewpointChange/)
add_subdirectory (ObstacleDetector/)

 
IF( ENABLE_OPENCV ) 
add_subdirectory (DisparityMapping/)
ELSE (ENABLE_OPENCV)
 MESSAGE("DisparityMapping will not be compiled , OpenCV stuff missing? ( check ENABLE_OPENCV flag )..")
ENDIF( ENABLE_OPENCV )

