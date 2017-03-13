/** @file ogl_shader_pipeline_renderer.h
 *  @brief  This is the new way ( OpenGL3+ ) to render using a shader based pipeline..
 *  @bug    The shader based rendering is not support is not yet complete
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef OGL_SHADER_PIPELINE_RENDERER_H_INCLUDED
#define OGL_SHADER_PIPELINE_RENDERER_H_INCLUDED

void doOGLShaderDrawCalllist(
                              float * vertices ,       unsigned int numberOfVertices ,
                              float * normal ,         unsigned int numberOfNormals ,
                              float * textureCoords ,  unsigned int numberOfTextureCoords ,
                              float * colors ,         unsigned int numberOfColors ,
                              unsigned int * indices , unsigned int numberOfIndices
                             );

#endif // OGL_SHADER_PIPELINE_RENDERER_H_INCLUDED
