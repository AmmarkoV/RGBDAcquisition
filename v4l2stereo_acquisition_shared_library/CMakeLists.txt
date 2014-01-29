project( V4L2StereoAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
  
add_library(V4L2StereoAcquisition SHARED 
            ../v4l2_acquisition_shared_library/PixelFormatConversions.c
            ../v4l2_acquisition_shared_library/PixelFormatConversions.h
            ../v4l2_acquisition_shared_library/V4L2_c.c
            ../v4l2_acquisition_shared_library/V4L2_c.h
            ../v4l2_acquisition_shared_library/V4L2Acquisition.c
            ../v4l2_acquisition_shared_library/V4L2Acquisition.h
            ../v4l2_acquisition_shared_library/V4L2IntrinsicCalibration.c
            ../v4l2_acquisition_shared_library/V4L2IntrinsicCalibration.h
            ../v4l2_acquisition_shared_library/V4L2Wrapper.c
            ../v4l2_acquisition_shared_library/V4L2Wrapper.h 
            V4L2StereoAcquisition.c
            V4L2StereoAcquisition.h
           ) 
target_link_libraries(V4L2StereoAcquisition rt CalibrationLibrary ) 
 
 

set_target_properties(V4L2StereoAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
