cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberCV)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../shared/cmake/Modules/")

find_package(DepthSense REQUIRED)
include_directories(${DEPTHSENSE_INCLUDE_DIR})

find_package(OpenCV REQUIRED)
#FIND_PACKAGE(OpenCV 2 REQUIRED)

add_executable (DepthSenseGrabberCV
	DepthSenseGrabberCV.cxx
	DepthSenseGrabberCV.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
)
message("test")
message(${OpenCV_LIBS})
target_link_libraries (DepthSenseGrabberCV ${DEPTHSENSESDK_LIBRARIES} ${OpenCV_LIBS})

#target_link_libraries (DepthSenseGrabberCV ${DEPTHSENSESDK_LIBRARIES})
