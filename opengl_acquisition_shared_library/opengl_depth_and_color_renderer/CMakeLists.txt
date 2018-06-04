project( OGLDepthAndColorRendererProject ) 
cmake_minimum_required( VERSION 2.8.7 )

set(CMAKE_CXX_FLAGS "-fPIC ")
set(CMAKE_C_FLAGS "-fPIC")

OPTION (ENABLE_HIGH_PERFORMANCE_BUILD OFF)
OPTION (ENABLE_JPG ON)
OPTION (ENABLE_PNG ON)
OPTION (ENABLE_SHADERS ON)
OPTION (ENABLE_ASSIMP OFF)

IF( ENABLE_HIGH_PERFORMANCE_BUILD )
 set(CMAKE_CXX_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
 set(CMAKE_C_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
ENDIF( ENABLE_HIGH_PERFORMANCE_BUILD )
      


#Add the codecs needed for the correct building
set(CODECS_DIRECTORY "../../tools/Codecs") 
add_subdirectory ( ${CODECS_DIRECTORY} RGBDAcquisition/Codecs)

add_subdirectory (src/)


IF( ENABLE_ASSIMP )
add_subdirectory (submodules/Assimp)
ENDIF( ENABLE_ASSIMP )

#First to make our handy tools
add_subdirectory (src/GameEngine/)
add_subdirectory (src/Renderer/)
add_subdirectory (src/Comparer/)
add_subdirectory (src/ModelDump/)
add_subdirectory (src/ModelMotionAnalysis/)
 
 

# TODO FIX INSTALLATION DIRECTORIES 

#    install(TARGETS RGBDAcquisitionProject
#                    LIBRARY DESTINATION lib
#                    ARCHIVE DESTINATION lib
#                    RUNTIME DESTINATION bin)
 

