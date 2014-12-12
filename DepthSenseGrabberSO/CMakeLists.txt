cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberSO)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../shared/cmake/Modules/")

find_package(DepthSense REQUIRED)
include_directories(${DEPTHSENSE_INCLUDE_DIR})

add_executable (DepthSenseGrabberSO
	DepthSenseGrabberSO.cxx
	DepthSenseGrabberSO.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
)
message(${DEPTHSENSE_INCLUDE_DIR})

target_link_libraries (DepthSenseGrabberSO ${DEPTHSENSE_LIBRARIES})
