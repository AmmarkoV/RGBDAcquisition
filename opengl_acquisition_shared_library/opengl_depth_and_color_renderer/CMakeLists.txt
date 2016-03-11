project( OGLDepthAndColorRendererProject ) 
cmake_minimum_required( VERSION 2.8.7 )


IF( COMPILE_FOR_PERFORMANCE )
 set(CMAKE_CXX_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
 set(CMAKE_C_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
ENDIF( COMPILE_FOR_PERFORMANCE )
      
add_subdirectory (src/)

#First to make our handy tools
add_subdirectory (src/Renderer/)
add_subdirectory (src/Comparer/)
add_subdirectory (src/ModelDump/)
 
 

# TODO FIX INSTALLATION DIRECTORIES 

#    install(TARGETS RGBDAcquisitionProject
#                    LIBRARY DESTINATION lib
#                    ARCHIVE DESTINATION lib
#                    RUNTIME DESTINATION bin)
 

