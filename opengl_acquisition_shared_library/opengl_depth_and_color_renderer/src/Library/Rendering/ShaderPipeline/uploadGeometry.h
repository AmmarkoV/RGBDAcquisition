#ifndef UPLOADGEOMETRY_H_INCLUDED
#define UPLOADGEOMETRY_H_INCLUDED



#include <GL/gl.h>


GLuint
pushBonesToBufferData(
                       int generateNewVao,
                        GLuint *vao ,
                        GLuint *arrayBuffer,
                        GLuint *elementBuffer,
                        GLuint programID  ,
                        const float * vertices , unsigned int sizeOfVertices ,
                        const float * normals , unsigned int sizeOfNormals ,
                        const float * textureCoords ,  unsigned int sizeOfTextureCoords ,
                        const float * colors , unsigned int sizeOfColors,
                        const unsigned int * indices , unsigned int sizeOfIndices
                     );

GLuint
pushObjectToBufferData(
                             int generateNewVao,
                             GLuint *vao,
                             GLuint *arrayBuffer,
                             GLuint *elementBuffer,
                             GLuint programID,
                             const float * vertices , unsigned int sizeOfVertices,
                             const float * normals , unsigned int sizeOfNormals,
                             const float * textureCoords ,  unsigned int sizeOfTextureCoords,
                             const float * colors , unsigned int sizeOfColors,
                             const unsigned int * indices , unsigned int sizeOfIndices
                      );

#endif // UPLOADGEOMETRY_H_INCLUDED
