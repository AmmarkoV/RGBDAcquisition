# - Try to find libfreenect
# Once done, this will define
#
# Freenect_FOUND - system has libfreenect
# Freenect_INCLUDE_DIRS - the libfreenect include directories
# Freenect_LIBRARIES - link these to use libfreenect

include(LibFindMacros)

# Use pkg-config to get hints about paths
libfind_pkg_check_modules(Freenect_PKGCONF libfreenect)

IF(NOT FREENECT_ROOT)
IF(EXISTS "/usr/include/libfreenect")
SET(FREENECT_ROOT "/usr")
ELSEIF(EXISTS "/usr/local/include/libfreenect")
SET(FREENECT_ROOT "/usr/local")
ELSE()
MESSAGE("FREENECT_ROOT not set. Continuing anyway..")
ENDIF()
ENDIF()

# Include dir
find_path(Freenect_INCLUDE_DIR
NAMES libfreenect.h
PATHS ${FREENECT_ROOT}/include/libfreenect ${Freenect_PKGCONF_INCLUDE_DIRS}
)

# Finally the library itself
find_library(Freenect_LIBRARY
NAMES freenect
PATHS ${FREENECT_ROOT}/lib ${Freenect_PKGCONF_LIBRARY_DIRS}
)

find_library(FreenectSync_LIBRARY
NAMES freenect_sync
PATHS ${FREENECT_ROOT}/lib ${Freenect_PKGCONF_LIBRARY_DIRS}
)

# Set the include dir variables and the libraries and let libfind_process do the rest.
# NOTE: Singular variables for this library, plural for libraries this this lib depends on.
set(Freenect_PROCESS_INCLUDES Freenect_INCLUDE_DIR Freenect_INCLUDE_DIRS)
set(Freenect_PROCESS_LIBS FreenectSync_LIBRARY Freenect_LIBRARY Freenect_LIBRARIES)
libfind_process(Freenect)
