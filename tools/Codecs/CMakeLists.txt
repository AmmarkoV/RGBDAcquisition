project( Codecs ) 
cmake_minimum_required( VERSION 2.8.7 )   
 
add_library(Codecs STATIC ppmInput.c ppmInput.h pngInput.c pngInput.h jpgInput.c jpgInput.h jpgExifexternal.c jpgExifexternal.h jpgExiforient_embed.c jpgExiforient_embed.h codecs.c codecs.h ) 
target_link_libraries(Codecs m rt ) 
 

set_target_properties(Codecs PROPERTIES 
                       DEBUG_POSTFIX "D"
                       ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                       RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
                      )
