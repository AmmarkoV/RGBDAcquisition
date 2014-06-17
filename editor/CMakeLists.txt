project( Editor ) 
cmake_minimum_required( VERSION 2.8.7 )  

add_executable(Editor  
               EditorApp.cpp
               EditorApp.h
               EditorMain.cpp
               EditorMain.h
               FeedScreenMemory.cpp
               FeedScreenMemory.h
               GetExtrinsics.cpp
               GetExtrinsics.h  
               SelectAcquisitionGraph.cpp
               SelectAcquisitionGraph.h 
               SelectCalibration.cpp
               SelectCalibration.h
               SelectModule.cpp
               SelectModule.h
               SelectSegmentation.cpp
               SelectSegmentation.h
               SelectTarget.cpp
               SelectTarget.h
               AddNewElement.cpp
               Tools.cpp
               Tools.h 
              ) 
 

 SET(wxWidgets_USE_LIBS)
 FIND_PACKAGE(wxWidgets)
 IF(wxWidgets_FOUND)
   INCLUDE("${wxWidgets_USE_FILE}") 
   target_link_libraries(Editor Acquisition AcquisitionMux AcquisitionSegment CalibrationLibrary ViewpointChange Quasirandomness OperatingSystem Timers rt dl m ${wxWidgets_LIBRARIES} )
   # and for each of your dependant executable/library targets: 
 ELSE(wxWidgets_FOUND)
   # For convenience. When we cannot continue, inform the user
   MESSAGE("wxWidgets not found!")
 ENDIF(wxWidgets_FOUND)


set_target_properties(Editor PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )


#    install(TARGETS Editor
#                    LIBRARY DESTINATION lib
#                    ARCHIVE DESTINATION lib
#                    RUNTIME DESTINATION bin)

