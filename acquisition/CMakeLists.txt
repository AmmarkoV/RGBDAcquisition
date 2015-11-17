project( Acquisition ) 
cmake_minimum_required( VERSION 2.8.7 )   
 
IF( ENABLE_LOCATION_SERVICE )
 MESSAGE("Location Service Will Be included in Acquisition")
 set(LocationService_Libs LocationServices gps ) 
 add_definitions(-DENABLE_LOCATION_SERVICE)
ENDIF( ENABLE_LOCATION_SERVICE )
   
    set(ACQUISITION_INGREDIENTS
                 Acquisition.c
                 Acquisition.h
                 pluginLinker.c
                 pluginLinker.h
                 processorLinker.c
                 processorLinker.h 
                 acquisition_setup.h
              )
  
    add_library(Acquisition SHARED ${ACQUISITION_INGREDIENTS} ) 
    target_link_libraries(Acquisition rt dl m Timers CalibrationLibrary OperatingSystem ${LocationService_Libs} ) 
 
    set_target_properties(Acquisition PROPERTIES 
                          DEBUG_POSTFIX "D"
                          ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                         )



    add_library(RGBDAcquisition SHARED ${ACQUISITION_INGREDIENTS} ) 
    target_link_libraries(RGBDAcquisition rt dl m Timers CalibrationLibrary OperatingSystem ) 
    set_target_properties(RGBDAcquisition PROPERTIES 
                          DEBUG_POSTFIX "D"
                          ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                         )
 
 
