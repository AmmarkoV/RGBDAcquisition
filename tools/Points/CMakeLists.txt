project( PointsDistance ) 
cmake_minimum_required( VERSION 2.8.7 )  

add_executable(PointsDistance pointsDistance.c  ) 
target_link_libraries(PointsDistance rt  m ) 
 

set_target_properties(PointsDistance PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
