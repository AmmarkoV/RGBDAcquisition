cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberCore)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../shared/cmake/Modules/")

find_package(DepthSense REQUIRED)
include_directories(${DEPTHSENSE_INCLUDE_DIR})
find_package(Boost 1.46.0 REQUIRED)

add_library (DepthSenseGrabberCore SHARED
	DepthSenseGrabberCore.cxx
	DepthSenseGrabberCore.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
)
message(${DEPTHSENSE_INCLUDE_DIR})

target_link_libraries (DepthSenseGrabberCore ${DEPTHSENSE_LIBRARIES} boost_system boost_thread)
