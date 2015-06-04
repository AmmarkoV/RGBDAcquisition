project( Freenect2Acquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
 

IF(ENABLE_CXX11)
  INCLUDE(CheckCXXCompilerFlag)
  CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
  CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
  IF(COMPILER_SUPPORTS_CXX11)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  ELSEIF(COMPILER_SUPPORTS_CXX0X)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
  ELSE()
    MESSAGE(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
  ENDIF()
ENDIF(ENABLE_CXX11)


SET(FREENECT_2_CENTRAL_DIR ${libfreenect2_SOURCE_DIR})
INCLUDE_DIRECTORIES("${FREENECT_2_CENTRAL_DIR}/include")

INCLUDE_DIRECTORIES(${LIBFREENECT2_THREADING_INCLUDE_DIR})

add_library(Freenect2Acquisition SHARED Freenect2Acquisition.cpp Freenect2Acquisition.h  ) 
target_link_libraries(Freenect2Acquisition rt dl m Timers CalibrationLibrary OperatingSystem  freenect2 )  
 

set_target_properties(Freenect2Acquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
