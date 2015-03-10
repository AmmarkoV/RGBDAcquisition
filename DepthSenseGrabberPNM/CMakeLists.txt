cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberPNM)

add_executable (DepthSenseGrabberPNM
	DepthSenseGrabberPNM.cxx
	DepthSenseGrabberPNM.hxx
	../DepthSenseGrabberCore/DepthSenseGrabberCore.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
)

target_link_libraries (DepthSenseGrabberPNM DepthSenseGrabberCore)
