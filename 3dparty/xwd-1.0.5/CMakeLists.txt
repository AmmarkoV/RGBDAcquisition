project( xwd ) 
cmake_minimum_required( VERSION 2.8.7 )

#Server applications need all the protection we can get..!
#set(CMAKE_C_FLAGS "-fPIC -fPIE -pie -s -Wl,-z,relro,-z,now -fstack-protector -O2")
set(CMAKE_C_FLAGS "-fPIC")

# Extend cmake's module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../cmake/modules ${CMAKE_MODULE_PATH})
   

set(XWD_INGREDIENTS clientwin.c clientwin.h dsimple.c dsimple.h list.c list.h multiVis.c multiVis.h config.h wsutils.h )
   
add_library(XwdLib STATIC ${XWD_INGREDIENTS} main.c )
add_executable(xwd ${XWD_INGREDIENTS} xwd.c ) 
target_link_libraries(xwd X11 )


       
set_target_properties(xwd PROPERTIES  
                          ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                         )
 
set_target_properties(XwdLib PROPERTIES  
                          ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                          RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                         )

