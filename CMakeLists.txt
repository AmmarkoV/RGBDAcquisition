cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabber)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include (GenerateExportHeader)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/shared/cmake/Modules/")
find_package(DepthSense REQUIRED)
include_directories(${DEPTHSENSE_INCLUDE_DIR})

find_package(Boost 1.46.0 COMPONENTS system thread REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

add_subdirectory(${CMAKE_SOURCE_DIR}/DepthSenseGrabberCore/)
add_subdirectory(${CMAKE_SOURCE_DIR}/DepthSenseGrabberPNM/)

find_package(OpenCV 2.4.10 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(${CMAKE_SOURCE_DIR}/DepthSenseGrabberOpenCV/)
