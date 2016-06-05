#ifndef MODEL_LOADER_TRI_H_INCLUDED
#define MODEL_LOADER_TRI_H_INCLUDED

#include "model_loader_setup.h"

#if HAVE_OBJ_CODE_AVAILIABLE
 #include "model_loader_obj.h"
#endif // HAVE_OBJ_CODE_AVAILIABLE

#define TRI_LOADER_VERSION 3

struct TRI_Header
{
     unsigned int triType;
     unsigned int floatSize;
     unsigned int drawType; //0 = triangles
     unsigned int numberOfVertices;
     unsigned int numberOfNormals;
     unsigned int numberOfTextureCoords;
     unsigned int numberOfColors;
     unsigned int numberOfIndices;
     //In order not to break this file format ever again
     unsigned int notUsed1;
     unsigned int notUsed2;
     unsigned int notUsed3;
     unsigned int notUsed4;
};


struct TRI_Model
{
   struct TRI_Header header;
   float * vertices;
   float * normal;
   float * textureCoords;
   float * colors;
   float * indices;
};



#if HAVE_OBJ_CODE_AVAILIABLE
int convertObjToTri(struct TRI_Model * tri , struct OBJ_Model * obj);
#endif // HAVE_OBJ_CODE_AVAILIABLE

int loadModelTri(const char * filename , struct TRI_Model * triModel);
int saveModelTri(const char * filename , struct TRI_Model * triModel);
//int saveModelTriHeader(const char * filename , struct TRI_Model * triModel);



#endif // MODEL_LOADER_TRI_H_INCLUDED
