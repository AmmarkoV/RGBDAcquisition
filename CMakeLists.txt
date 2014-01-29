project( RGBDAcquisitionProject ) 
cmake_minimum_required( VERSION 2.8.7 )

set(CMAKE_CXX_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
set(CMAKE_C_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
     
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
add_subdirectory (v4l2_acquisition_shared_library/)
add_subdirectory (v4l2stereo_acquisition_shared_library/)
#THIS DOESNT WORK YET > add_subdirectory (libfreenect_acquisition_shared_library/)



#Composite ( bloated ;p ) parts 
add_subdirectory (editor/)
add_subdirectory (viewer/)

# TODO FIX INSTALLATION DIRECTORIES 

#    install(TARGETS RGBDAcquisitionProject
#                    LIBRARY DESTINATION lib
#                    ARCHIVE DESTINATION lib
#                    RUNTIME DESTINATION bin)
 

