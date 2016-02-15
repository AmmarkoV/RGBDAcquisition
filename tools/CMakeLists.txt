project( Tools ) 
cmake_minimum_required( VERSION 2.8.7 ) 
add_subdirectory (Timers/)
add_subdirectory (Calibration/)
add_subdirectory (OperatingSystem/)
add_subdirectory (Quasirandomness/) 
add_subdirectory (Points/) 
add_subdirectory (Codecs/) 
add_subdirectory (AmMatrix/) 
add_subdirectory (Quaternions/)
add_subdirectory (ImageOperations/)
add_subdirectory (DepthImagesConverter/)



if (ENABLE_LOCATION_SERVICE)
 add_subdirectory (LocationServices/)
ENDIF(ENABLE_LOCATION_SERVICE)

IF( ENABLE_OPENCV )
  #add_subdirectory (DepthImagesConverter/) 
ENDIF( ENABLE_OPENCV )

