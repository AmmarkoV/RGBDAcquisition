project( OGLDepthAndColorRendererProject ) 
cmake_minimum_required( VERSION 2.8.7 )

set(CMAKE_CXX_FLAGS "-fPIC ")
set(CMAKE_C_FLAGS "-fPIC")

OPTION (ENABLE_HIGH_PERFORMANCE_BUILD OFF)
OPTION (ENABLE_JPG ON)
OPTION (ENABLE_PNG ON)
OPTION (ENABLE_SHADERS ON)
OPTION (ENABLE_ASSIMP OFF)
OPTION (ENABLE_AMMARSERVER OFF)

IF( ENABLE_HIGH_PERFORMANCE_BUILD )
 set(CMAKE_CXX_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
 set(CMAKE_C_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
ENDIF( ENABLE_HIGH_PERFORMANCE_BUILD )
      
IF( ENABLE_AMMARSERVER )
 add_subdirectory (submodules/AmmarServer/) 
 add_definitions(-DUSE_AMMARSERVER)

 execute_process(COMMAND submodules/linkAmmarServer.sh)
ENDIF( ENABLE_AMMARSERVER )

#Add the codecs needed for the correct building
set(CODECS_DIRECTORY "../../tools/Codecs") 
add_subdirectory ( ${CODECS_DIRECTORY} RGBDAcquisition/Codecs)

add_subdirectory (src/)


IF( ENABLE_ASSIMP )
add_subdirectory (submodules/Assimp)
ENDIF( ENABLE_ASSIMP )

#First to make our handy tools
IF( ENABLE_SHADERS )
add_subdirectory (src/Applications/GameEngine/)
ENDIF( ENABLE_SHADERS )


add_subdirectory (src/Applications/Renderer/)
add_subdirectory (src/Applications/Comparer/)
add_subdirectory (src/Applications/ModelDump/)
add_subdirectory (src/Applications/ModelMotionAnalysis/)
 
 

# TODO FIX INSTALLATION DIRECTORIES 

#    install(TARGETS RGBDAcquisitionProject
#                    LIBRARY DESTINATION lib
#                    ARCHIVE DESTINATION lib
#                    RUNTIME DESTINATION bin)
 

