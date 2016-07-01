project( DepthImagesConverter )
cmake_minimum_required( VERSION 2.8.7 )
 
find_package(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
 
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


IF( ENABLE_OPENCV )
 MESSAGE("OpenCV will be included in this DepthImageConverter build")
 set(OCV_Libs ${OpenCV_LIBRARIES} ) 
 add_definitions(-DUSE_OPENCV) 
ENDIF( ENABLE_OPENCV )


add_executable(DepthImagesConverter main.cpp ) 
target_link_libraries(DepthImagesConverter rt dl m Codecs ${PNG_Libs} ${JPG_Libs} ${OCV_Libs} )
set_target_properties(DepthImagesConverter PROPERTIES DEBUG_POSTFIX "D")
add_dependencies(DepthImagesConverter Codecs)  
       
 
set_target_properties(DepthImagesConverter PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )

 
