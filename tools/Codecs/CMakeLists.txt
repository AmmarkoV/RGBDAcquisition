project( Codecs )
cmake_minimum_required( VERSION 2.8.7 )


set(CMAKE_CXX_FLAGS "-s -O3 -fPIC -march=native -mtune=native") # -fuse-ld=gold
set(CMAKE_C_FLAGS "-s -O3 -fPIC -march=native -mtune=native")  # -fuse-ld=gold

IF( ENABLE_JPG )
 MESSAGE("JPGs will be included in this codec build")
 set(JPG_Libs jpeg ) 
 set(JPG_Parts jpgInput.c jpgInput.h jpgExifexternal.c jpgExifexternal.h jpgExiforient_embed.c jpgExiforient_embed.h )
 set(JPG_Includes ${CMAKE_SOURCE_DIR}/3dparty/OpenNI2/Include/ )
 add_definitions(-DUSE_JPG_FILES)
 add_definitions(-DENABLE_JPG)
ENDIF( ENABLE_JPG )


IF( ENABLE_PNG )
 MESSAGE("PNGs will be included in this codec build")
 set(PNG_Libs png ) 
 set(PNG_Parts pngInput.c pngInput.h)
 set(PNG_Includes ${CMAKE_SOURCE_DIR}/3dparty/OpenNI2/Include/ )
 add_definitions(-DUSE_PNG_FILES)
 add_definitions(-DENABLE_PNG)
ENDIF( ENABLE_PNG )


add_library(
            Codecs STATIC
            pfmInput.c
            pfmInput.h
            ppmInput.c
            ppmInput.h
            bmpInput.c
            bmpInput.h
            asciiInput.c
            asciiInput.h
            ${PNG_Parts}
            ${JPG_Parts}
            codecs.c
            codecs.h
           )
target_link_libraries(Codecs m rt )
 


add_executable(codecconverter utilConverterMain.c ) 
target_link_libraries(codecconverter rt dl m Codecs ${PNG_Libs} ${JPG_Libs} )
set_target_properties(codecconverter PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(codecconverter Codecs)  
       
 
set_target_properties(codecconverter PROPERTIES 
                       #DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )



set_target_properties(Codecs PROPERTIES
                       #DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
