###############################################################################
# Find DepthSense SDK
#
# This sets the following variables:
# DEPTHSENSE_FOUND - True if DEPTHSENSE was found.
# DEPTHSENSE_INCLUDE_DIR - Directories containing the DEPTHSENSE include files.
# DEPTHSENSE_LIBRARIES - Libraries needed to use DEPTHSENSE.

if (UNIX)
    set(PROGRAM_FILES_PATHS "/opt/softkinetic/DepthSenseSDK")
elseif( WIN32)
    if (CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(PROGRAM_FILES_PATHS "$ENV{PROGRAMW6432}/SoftKinetic/DepthSenseSDK")
    else ()
        set(PROGRAM_FILES_PATHS "$ENV{PROGRAMFILES}/SoftKinetic/DepthSenseSDK")
    endif()
endif()

find_path(DEPTHSENSE_INCLUDE_DIR DepthSense.hxx
          HINTS ${NESTK_ROOT_DIRS_HINTS} "${DEPTHSENSE_ROOT}" "$ENV{DEPTHSENSE_ROOT}"
          PATHS "${PROGRAM_FILES_PATHS}"
          PATH_SUFFIXES include)

find_library(DEPTHSENSE_LIBRARIES
           NAMES DepthSense
           HINTS ${NESTK_ROOT_DIRS_HINTS} "${DEPTHSENSE_ROOT}" "$ENV{DEPTHSENSE_ROOT}"
           PATHS "${PROGRAM_FILES_PATHS}"
           PATH_SUFFIXES lib )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DEPTHSENSE DEFAULT_MSG DEPTHSENSE_LIBRARIES DEPTHSENSE_INCLUDE_DIR)
mark_as_advanced(DEPTHSENSE_LIBRARIES DEPTHSENSE_INCLUDE_DIR)

if(DEPTHSENSE_FOUND)
    message(STATUS "DEPTHSENSE found (include: ${DEPTHSENSE_INCLUDE_DIR}, lib: ${DEPTHSENSE_LIBRARIES})")
endif(DEPTHSENSE_FOUND)
