cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabber)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/shared/cmake/Modules/")
find_package(DepthSense REQUIRED)
include_directories(${DEPTHSENSE_INCLUDE_DIR})
find_package(Boost 1.46.0 REQUIRED)

#include(${CMAKE_SOURCE_DIR}/DepthSenseGrabberCore/CMakeLists.txt)
add_subdirectory(${CMAKE_SOURCE_DIR}/DepthSenseGrabberCore/)
add_subdirectory(${CMAKE_SOURCE_DIR}/DepthSenseGrabberPNM/)
#include_directories(${CMAKE_SOURCE_DIR}/DepthSenseGrabberCore/)
#include_directories(${CMAKE_SOURCE_DIR}/DepthSenseGrabberPNM/)

#find_package(DepthSense REQUIRED)
#include_directories(${DEPTHSENSE_INCLUDE_DIR})
#find_package(Boost 1.46.0 REQUIRED)

#add_executable (DepthSenseGrabberPNM
#	DepthSenseGrabberPNM.cxx
#	DepthSenseGrabberPNM.hxx
#	../DepthSenseGrabberCore/DepthSenseGrabberCore.cxx
#	../DepthSenseGrabberCore/DepthSenseGrabberCore.hxx
#	../shared/ConversionTools.cxx
#	../shared/ConversionTools.hxx
#)
#message(${DEPTHSENSE_INCLUDE_DIR})

#target_link_libraries (DepthSenseGrabberPNM ${DEPTHSENSE_LIBRARIES} boost_system boost_thread)
