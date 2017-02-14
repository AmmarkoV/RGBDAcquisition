#ifndef OGL_FIXED_PIPELINE_RENDERER_H_INCLUDED
#define OGL_FIXED_PIPELINE_RENDERER_H_INCLUDED


void doOGLBoneDrawCalllist( float * pos , unsigned int * parentNode , unsigned int boneSizes);

void doOGLGenericDrawCalllist(
                              float * vertices ,       unsigned int numberOfVertices ,
                              float * normal ,         unsigned int numberOfNormals ,
                              float * textureCoords ,  unsigned int numberOfTextureCoords ,
                              float * colors ,         unsigned int numberOfColors ,
                              unsigned int * indices , unsigned int numberOfIndices
                             );

#endif // OGL_FIXED_PIPELINE_RENDERER_H_INCLUDED
