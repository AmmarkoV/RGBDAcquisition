project( OGLDepthAndColorRendererProject ) 
cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "-fPIC -march=native")
set(CMAKE_C_FLAGS "-fPIC -march=native")

OPTION (ENABLE_HIGH_PERFORMANCE_BUILD OFF)
OPTION (ENABLE_JPG ON)
OPTION (ENABLE_PNG ON)
OPTION (ENABLE_SHADERS ON)
OPTION (ENABLE_ASSIMP OFF)
OPTION (ENABLE_AMMARSERVER OFF)

OPTION(INTEL_OPTIMIZATIONS OFF)

if (INTEL_OPTIMIZATIONS)
add_definitions(-DINTEL_OPTIMIZATIONS)
endif(INTEL_OPTIMIZATIONS)


#Add the codecs needed for the correct building..
#Depending on the build directory this might not work :S
set(CODECS_DIRECTORY "../../tools/Codecs") 
add_subdirectory ( ${CODECS_DIRECTORY} RGBDAcquisition/tools/Codecs) 
#ADD_LIBRARY(Codecs STATIC IMPORTED) 
#SET_TARGET_PROPERTIES(Codecs PROPERTIES IMPORTED_LOCATION ${CODECS_DIRECTORY}/libCodecs.a) 


IF( ENABLE_HIGH_PERFORMANCE_BUILD )
 set(CMAKE_CXX_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
 set(CMAKE_C_FLAGS "-fPIC -march=native -mtune=native -O3 -fexpensive-optimizations -s") 
ENDIF( ENABLE_HIGH_PERFORMANCE_BUILD )
      
IF( ENABLE_AMMARSERVER )
 add_subdirectory (submodules/AmmarServer/) 
 add_definitions(-DUSE_AMMARSERVER)

 execute_process(COMMAND submodules/linkAmmarServer.sh)
ENDIF( ENABLE_AMMARSERVER )

add_subdirectory (src/Library)


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
add_subdirectory (src/Applications/BVHTester/)
 
 

# TODO FIX INSTALLATION DIRECTORIES 

#    install(TARGETS RGBDAcquisitionProject
#                    LIBRARY DESTINATION lib
#                    ARCHIVE DESTINATION lib
#                    RUNTIME DESTINATION bin)
 

