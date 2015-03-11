cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberOpenCV)

add_executable (DepthSenseGrabberOpenCV
	DepthSenseGrabberOpenCV.cxx
	DepthSenseGrabberOpenCV.hxx
	../DepthSenseGrabberCore/DepthSenseGrabberCore.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
)

target_link_libraries (DepthSenseGrabberOpenCV DepthSenseGrabberCore ${OpenCV_LIBS} boost_system)
