cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberSO)

set(DEPTHSENSESDK_DIR /opt/softkinetic/DepthSenseSDK)
set(DEPTHSENSESDK_INCLUDE_DIR ${DEPTHSENSESDK_DIR}/include)
set(DEPTHSENSESDK_LIBRARY_DIR ${DEPTHSENSESDK_DIR}/lib)
set(DEPTHSENSESDK_LIBRARIES ${DEPTHSENSESDK_LIBRARY_DIR}/libDepthSense.so ${DEPTHSENSESDK_LIBRARY_DIR}/libDepthSensePlugins.so ${DEPTHSENSESDK_LIBRARY_DIR}/libturbojpeg.so ${DEPTHSENSESDK_LIBRARY_DIR}/libusb-1.0.so)

include_directories(${DEPTHSENSESDK_INCLUDE_DIR})
link_directories(${DEPTHSENSESDK_LIBRARY_DIR})

add_executable (DepthSenseGrabberSO
	DepthSenseGrabberSO.cxx
	DepthSenseGrabberSO.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
)
target_link_libraries (DepthSenseGrabberSO ${DEPTHSENSESDK_LIBRARIES})
