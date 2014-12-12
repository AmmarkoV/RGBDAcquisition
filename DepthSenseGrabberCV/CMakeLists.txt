cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberCV)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/../shared/cmake/Modules/")

find_package(DepthSense REQUIRED)
include_directories(${DEPTHSENSE_INCLUDE_DIR})

find_package(OpenCV 2.4.10 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable (DepthSenseGrabberCV
	DepthSenseGrabberCV.cxx
	DepthSenseGrabberCV.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
)
message("test")
message(${OpenCV_LIBS})
message(${DEPTHSENSE_LIBRARIES})
message(${DEPTHSENSE_INCLUDE_DIR})
target_link_libraries (DepthSenseGrabberCV ${DEPTHSENSE_LIBRARIES} ${OpenCV_LIBS})
