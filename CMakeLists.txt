cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabber)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include (GenerateExportHeader)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/shared/cmake/Modules/")
find_package(DepthSense REQUIRED)
include_directories(${DEPTHSENSE_INCLUDE_DIR})

#find_package(Boost 1.46.0 REQUIRED)
find_package(Boost 1.46.0 REQUIRED COMPONENTS system thread)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS}/system)
link_directories(${Boost_INCLUDE_DIRS}/lib64-msvc-12.0)
message(${Boost_INCLUDE_DIRS})
message(${Boost_LIBRARY_DIRS}/system)
message(${Boost_LIBRARIES}/test)
message(${Boost_INCLUDEDIR}/test)
message(${Boost_FOUND}/test)
message(${Boost_VERSION}/test)

add_subdirectory(${CMAKE_SOURCE_DIR}/DepthSenseGrabberCore/)
add_subdirectory(${CMAKE_SOURCE_DIR}/DepthSenseGrabberPNM/)

find_package(OpenCV 2.4.10 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(${CMAKE_SOURCE_DIR}/DepthSenseGrabberOpenCV/)
