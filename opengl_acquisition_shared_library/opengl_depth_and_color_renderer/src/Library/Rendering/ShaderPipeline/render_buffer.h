#ifndef RENDER_BUFFER_H_INCLUDED
#define RENDER_BUFFER_H_INCLUDED


#include <GL/gl.h>

#include "../../../../../../tools/AmMatrix/matrix4x4Tools.h"


int initializeFramebuffer(
                          GLuint * FramebufferName,
                          GLuint * renderedTexture,
                          GLuint * depthTexture,
                          unsigned int width,
                          unsigned int height
                         );

int drawFramebufferTexToTex(
                       GLuint FramebufferName,
                       GLuint programFrameBufferID,
                       GLuint quad_vertexbuffer,
                       GLuint renderedTexture,
                       GLuint texID,
                       GLuint timeID,
                       GLuint resolutionID,
                       unsigned int width,
                       unsigned int height
                   );

int drawFramebufferToScreen(
                      // GLuint FramebufferName,
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
                               GLuint FramebufferName,
                               GLuint textureToDraw,
                               GLuint programFrameBufferID,
                               GLuint quad_vertexbuffer,
                               GLuint texID,
                               GLuint timeID,
                               GLuint resolutionID,
                               unsigned int width,
                               unsigned int height
                              );


int drawVertexArrayWithMVPMatrices(
                                   GLuint programID,
                                   GLuint vao,
                                   GLuint MatrixID,
                                   GLuint TextureID,
                                   unsigned int triangleCount,
                                   unsigned int elementCount,
                                   //-----------------------------------------
                                   struct Matrix4x4OfFloats * modelMatrix,
                                   //-----------------------------------------
                                   struct Matrix4x4OfFloats * projectionMatrix,
                                   struct Matrix4x4OfFloats * viewportMatrix,
                                   struct Matrix4x4OfFloats * viewMatrix,
                                   //-----------------------------------------
                                   char renderWireframe
                                  );

#endif // RENDER_BUFFER_H_INCLUDED
