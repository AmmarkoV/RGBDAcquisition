project( FreenectAcquisition ) 
cmake_minimum_required( VERSION 2.8.7 )  
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
  


set(FREENECT_DEFINITIONS ${PC_FREENECT_CFLAGS_OTHER})
FIND_LIBRARY( FREENECT_LIBRARY
             NAMES OpenNI2
             HINTS ${PC_FREENECT_LIBDIR} ${PC_FREENECT_LIBRARY_DIRS} /usr/lib
             PATHS "$ENV{PROGRAMFILES}/OpenNI2/Lib${FREENECT_SUFFIX}" "$ENV{PROGRAMW6432}/OpenNI2/Lib${FREENECT_SUFFIX}" "$ENV{PROGRAMW6432}/OpenNI2"
             PATH_SUFFIXES lib lib64
)
FIND_PATH( FREENECT_INCLUDE_DIR OpenNI.h
          HINTS ${PC_FREENECT_INCLUDEDIR} ${PC_FREENECT_INCLUDE_DIRS} 
                  /usr/include/openni2 /usr/include/ni2
                  PATHS "$ENV{PROGRAMFILES}/OpenNI2/include" "$ENV{PROGRAMW6432}/OpenNI2/include"
          PATH_SUFFIXES openni2 ni2)
 



    add_library(FreenectAcquisition SHARED FreenectAcquisition.c FreenectAcquisition.h ) 
    target_link_libraries(FreenectAcquisition rt dl m Timers CalibrationLibrary OperatingSystem ) 
 #TODO THIS IS NOT WORKING
 

set_target_properties(FreenectAcquisition PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
