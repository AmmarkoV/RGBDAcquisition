#ifndef RENDER_BUFFER_H_INCLUDED
#define RENDER_BUFFER_H_INCLUDED


#include <GL/gl.h>

int initializeFramebuffer(
                          GLuint * FramebufferName,
                          GLuint * renderedTexture,
                          GLuint * depthTexture,
                          unsigned int width,
                          unsigned int height
                         );

int drawFramebuffer(
                       GLuint programFrameBufferID,
                       GLuint quad_vertexbuffer,
                       GLuint renderedTexture,
                       GLuint texID,
                       GLuint timeID,
                       GLuint resolutionID,
                       unsigned int width,
                       unsigned int height
                   );



int drawFramebufferFromTexture(
                               GLuint textureToDraw,
                       GLuint programFrameBufferID,
                       GLuint quad_vertexbuffer,
                       GLuint renderedTexture,
                       GLuint texID,
                       GLuint timeID,
                       GLuint resolutionID,
                       unsigned int width,
                       unsigned int height
                   );

#endif // RENDER_BUFFER_H_INCLUDED
