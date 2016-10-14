#ifndef OGL_RENDERING_H_INCLUDED
#define OGL_RENDERING_H_INCLUDED



int startOGLRendering();

int renderOGL(
               float * vertices ,       unsigned int numberOfVertices ,
               float * normal ,         unsigned int numberOfNormals ,
               float * textureCoords ,  unsigned int numberOfTextureCoords ,
               float * colors ,         unsigned int numberOfColors ,
               unsigned int * indices , unsigned int numberOfIndices
             );

int stopOGLRendering();

#endif // OGL_RENDERING_H_INCLUDED
