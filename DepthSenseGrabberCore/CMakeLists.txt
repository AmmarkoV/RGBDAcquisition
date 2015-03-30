cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(DepthSenseGrabberCore)

#add_library (DepthSenseGrabberCore SHARED
add_library (DepthSenseGrabberCore
	DepthSenseGrabberCore.cxx
	DepthSenseGrabberCore.hxx
	../shared/ConversionTools.cxx
	../shared/ConversionTools.hxx
	../shared/AcquisitionParameters.hxx
)

GENERATE_EXPORT_HEADER( DepthSenseGrabberCore
             BASE_NAME DepthSenseGrabberCore
             EXPORT_MACRO_NAME DepthSenseGrabberCore_EXPORT
             EXPORT_FILE_NAME DepthSenseGrabberCore_Export.h
             STATIC_DEFINE DepthSenseGrabberCore_BUILT_AS_STATIC
)

#target_link_libraries (DepthSenseGrabberCore ${DEPTHSENSE_LIBRARIES} boost_system boost_thread)
target_link_libraries (DepthSenseGrabberCore ${DEPTHSENSE_LIBRARIES})
