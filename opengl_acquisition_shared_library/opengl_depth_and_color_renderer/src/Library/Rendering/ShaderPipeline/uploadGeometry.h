#ifndef UPLOADGEOMETRY_H_INCLUDED
#define UPLOADGEOMETRY_H_INCLUDED

#include <GL/gl.h>

struct shaderModelData
{
  unsigned long timestamp;

  //OpenGL descriptors
  char   initialized;
  GLuint VAO;
  GLuint arrayBuffer;
  GLuint elementBuffer;
  unsigned int triangleCount;

  //Base geometry
  unsigned long lastTimestampBaseModification;
  float * vertices;       unsigned int sizeOfVertices;
  float * normals;        unsigned int sizeOfNormals;
  float * textureCoords;  unsigned int sizeOfTextureCoords;
  float * colors;         unsigned int sizeOfColors;
  unsigned int * indices; unsigned int sizeOfIndices;
  //-------------------------------------------------------------------
  unsigned long lastTimestampBoneModification;
  unsigned int numberOfBones;
  unsigned int numberOfBonesPerVertex;
  unsigned int * boneIndexes;  unsigned int sizeOfBoneIndexes;
  float * boneWeightValues;    unsigned int sizeOfBoneWeightValues;
  float * boneTransforms;      unsigned int sizeOfBoneTransforms;
  //-------------------------------------------------------------------

  //Pointer to The (TRI) Model Structure
  void * model;
};



GLuint
pushBonesToBufferData(
                        int generateNewVao,
                        int generateNewArrayBuffer,
                        int generateNewElementBuffer,
                        GLuint programID,
                        //-------------------------------------------------------------------
                        struct shaderModelData * shaderData
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
