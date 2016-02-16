cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberPCL)

find_package(PCL 1.2 REQUIRED)

set(DEPTHSENSESDK_DIR /opt/softkinetic/DepthSenseSDK)
set(DEPTHSENSESDK_INCLUDE_DIR ${DEPTHSENSESDK_DIR}/include)
set(DEPTHSENSESDK_LIBRARY_DIR ${DEPTHSENSESDK_DIR}/lib)
set(DEPTHSENSESDK_LIBRARIES ${DEPTHSENSESDK_LIBRARY_DIR}/libDepthSense.so ${DEPTHSENSESDK_LIBRARY_DIR}/libDepthSensePlugins.so ${DEPTHSENSESDK_LIBRARY_DIR}/libturbojpeg.so ${DEPTHSENSESDK_LIBRARY_DIR}/libusb-1.0.so)

include_directories(${PCL_INCLUDE_DIRS} ${DEPTHSENSESDK_INCLUDE_DIR})
link_directories(${PCL_LIBRARY_DIRS} ${DEPTHSENSESDK_LIBRARY_DIR})
add_definitions(${PCL_DEFINITIONS})

add_executable (DepthSenseGrabberPCL
	DepthSenseGrabberPCL.cxx
	DepthSenseGrabberPCL.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
)
target_link_libraries (DepthSenseGrabberPCL ${PCL_LIBRARIES} ${DEPTHSENSESDK_LIBRARIES})
