/** @file downloadFromRenderer.h
 *  @brief  This is file should handle all the OpenGL drawing and be able to switch graphics output to use fixed or shader based pipelines
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef OGL_DOWNLOADFROMRENDERER_H_INCLUDED
#define OGL_DOWNLOADFROMRENDERER_H_INCLUDED


int downloadOpenGLColor(unsigned char * color , unsigned int x,unsigned int y,unsigned int width,unsigned int height);

int downloadOpenGLZBuffer(unsigned short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height,float depthScale);

int downloadOpenGLDepth(unsigned short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height,float depthScale);



int downloadOpenGLColorFromTexture(const char * filename , unsigned int tex, unsigned int width , unsigned int height);
int downloadOpenGLDepthFromTexture(const char * filename , unsigned int tex , unsigned int width , unsigned int height);


#endif // OGL_DOWNLOADFROMRENDERER_H_INCLUDED

