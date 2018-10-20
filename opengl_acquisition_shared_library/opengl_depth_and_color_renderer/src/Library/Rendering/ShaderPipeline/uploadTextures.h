#ifndef UPLOADTEXTURES_H_INCLUDED
#define UPLOADTEXTURES_H_INCLUDED

#include <GL/gl.h>

int uploadColorImageAsTexture(
                               GLuint programID,
                               GLuint *textureID,
                               unsigned int * alreadyUploaded,
                               unsigned char * colorPixels,
                               unsigned int colorWidth ,
                               unsigned int colorHeight ,
                               unsigned int colorChannels ,
                               unsigned int colorBitsperpixel
                              );


int uploadDepthImageAsTexture(
                               GLuint programID  ,
                               GLuint *textureID,
                               unsigned int * alreadyUploaded,
                               unsigned short * depthPixels,
                               unsigned int depthWidth,
                               unsigned int depthHeight,
                               unsigned int depthChannels,
                               unsigned int depthBitsPerPixel
                              );

#endif // UPLOADTEXTURES_H_INCLUDED
