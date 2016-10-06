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
