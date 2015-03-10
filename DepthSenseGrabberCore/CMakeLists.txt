cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberCore)

add_library (DepthSenseGrabberCore SHARED
	DepthSenseGrabberCore.cxx
	DepthSenseGrabberCore.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
	../shared/AcquisitionParameters.hxx
)

target_link_libraries (DepthSenseGrabberCore ${DEPTHSENSE_LIBRARIES} boost_system boost_thread)
