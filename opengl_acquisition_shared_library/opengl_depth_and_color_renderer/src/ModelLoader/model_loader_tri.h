#ifndef MODEL_LOADER_TRI_H_INCLUDED
#define MODEL_LOADER_TRI_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif

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
     unsigned int numberOfVertices; // the number of vertices
     unsigned int numberOfNormals;
     unsigned int numberOfTextureCoords;
     unsigned int numberOfColors;
     unsigned int numberOfIndices;
     unsigned int numberOfBones;
     //In order not to break this file format ever again
     unsigned int notUsed1;
     unsigned int notUsed2;
     unsigned int notUsed3;
};




struct TRI_Bones_Header
{
  unsigned int boneParent;
  unsigned int boneWeightsNumber;
  unsigned int boneNameSize;
  float inverseBindPose[16];
};

struct TRI_Bones
{
  struct TRI_Bones_Header info;
  char*  boneName;
  float * weightValue;
  unsigned int * weightIndex;
};

struct TRI_Model
{
   struct TRI_Header header;
   float * vertices;
   float * normal;
   float * textureCoords;
   float * colors;
   struct TRI_Bones * bones;
   unsigned int * indices;
};



#if HAVE_OBJ_CODE_AVAILIABLE
int convertObjToTri(struct TRI_Model * tri , struct OBJ_Model * obj);
#endif // HAVE_OBJ_CODE_AVAILIABLE

int fillFlatModelTriFromIndexedModelTri(struct TRI_Model * triModel , struct TRI_Model * indexed);

struct TRI_Model * allocateModelTri();
int freeModelTri(struct TRI_Model * triModel);

int loadModelTri(const char * filename , struct TRI_Model * triModel);
int saveModelTri(const char * filename , struct TRI_Model * triModel);

void copyModelTri(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN );
void deallocModelTri(struct TRI_Model * triModel);

void doTriDrawCalllist(struct TRI_Model * tri );

//int saveModelTriHeader(const char * filename , struct TRI_Model * triModel);


#ifdef __cplusplus
}
#endif

#endif // MODEL_LOADER_TRI_H_INCLUDED
