project( DepthSenseAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
    
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/3dparty/")
find_package(DepthSense REQUIRED)
include_directories(${DEPTHSENSE_INCLUDE_DIR})
   
find_package(Boost 1.46.0 COMPONENTS system thread REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_library(DepthSenseAcquisition SHARED
	    ../3dparty/DepthSenseGrabber/DepthSenseGrabberCore/DepthSenseGrabberCore.cxx
	    ../3dparty/DepthSenseGrabber/DepthSenseGrabberCore/DepthSenseGrabberCore.hxx
	    ../3dparty/DepthSenseGrabber/shared/ConversionTools.cxx
	    ../3dparty/DepthSenseGrabber/shared/ConversionTools.hxx
            ../3dparty/DepthSenseGrabber/shared/AcquisitionParameters.hxx 


            DepthSenseAcquisition.cxx
            DepthSenseAcquisition.hxx
           ) 
target_link_libraries(DepthSenseAcquisition rt m CalibrationLibrary OperatingSystem Codecs  ${DEPTHSENSE_LIBRARIES} ${Boost_LIBRARIES} ) 
 
 

set_target_properties(DepthSenseAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
