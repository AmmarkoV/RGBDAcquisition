project( OpenCVBindings ) 
cmake_minimum_required( VERSION 2.8.7 )
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
 
 
add_executable(sift sift.cpp ) 
target_link_libraries(sift rt dl m  ${OpenCV_LIBRARIES} ) 
add_dependencies(sift Codecs   )  
 
 
set_target_properties(sift PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )

 
