project( RGBDAcquisitionProject ) 
cmake_minimum_required( VERSION 2.8.7 )

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/3dparty ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
set(CMAKE_C_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
      
OPTION (ENABLE_AMMARSERVER_BROADCAST ON)
OPTION (ENABLE_FREENECT ON)
OPTION (ENABLE_TEMPLATE ON)
OPTION (ENABLE_V4L2 ON)
OPTION (ENABLE_OPENNI1 ON)
OPTION (ENABLE_OPENNI2 ON)
OPTION (ENABLE_OPENGL ON)
OPTION (ENABLE_NETWORK ON)
OPTION (ENABLE_EDITOR ON)

add_subdirectory (3dparty/fast)

#First to make our handy tools
add_subdirectory (tools/)

#This is the core library
add_subdirectory (acquisition/)
add_subdirectory (acquisitionSegment/)
add_subdirectory (acquisition_mux/)
#This is the core application  
add_subdirectory (grabber/)
add_subdirectory (grabber_mux/)
add_subdirectory (grabber_segment/)
  

#Acquisition Plugins 
IF( ENABLE_TEMPLATE )
add_subdirectory (template_acquisition_shared_library/)
ENDIF( ENABLE_TEMPLATE )

IF( ENABLE_V4L2 )
add_subdirectory (v4l2_acquisition_shared_library/)
add_subdirectory (v4l2stereo_acquisition_shared_library/)
ENDIF( ENABLE_V4L2 )
 
IF( ENABLE_NETWORK )
  add_subdirectory (network_acquisition_shared_library/)
ENDIF( ENABLE_NETWORK )
 
IF( ENABLE_FREENECT )
  add_subdirectory (libfreenect_acquisition_shared_library/)
ENDIF( ENABLE_FREENECT )

IF( ENABLE_OPENNI1 )
  add_subdirectory (openni1_acquisition_shared_library/)
ENDIF( ENABLE_OPENNI1 )

IF( ENABLE_OPENNI2 )
  add_subdirectory (openni2_acquisition_shared_library/)
ENDIF( ENABLE_OPENNI2 )

IF( ENABLE_OPENGL )
  add_subdirectory (opengl_acquisition_shared_library/)
ENDIF( ENABLE_OPENGL )
 
 
#Composite ( bloated ;p ) parts 
IF( ENABLE_EDITOR )
add_subdirectory (editor/)
ENDIF( ENABLE_EDITOR )

add_subdirectory (viewer/)

# TODO FIX INSTALLATION DIRECTORIES 

#    install(TARGETS RGBDAcquisitionProject
#                    LIBRARY DESTINATION lib
#                    ARCHIVE DESTINATION lib
#                    RUNTIME DESTINATION bin)
 

