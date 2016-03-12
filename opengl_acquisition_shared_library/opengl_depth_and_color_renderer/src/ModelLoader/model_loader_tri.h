#ifndef MODEL_LOADER_TRI_H_INCLUDED
#define MODEL_LOADER_TRI_H_INCLUDED

#define HAVE_OBJ_CODE_AVAILIABLE  1

#if HAVE_OBJ_CODE_AVAILIABLE
 #include "model_loader_obj.h"
#endif // HAVE_OBJ_CODE_AVAILIABLE

struct TRI_Header
{
     unsigned int triType;
     unsigned int floatSize;
     unsigned int numberOfTriangles;
     unsigned int numberOfNormals;

};


struct TRI_Model
{
   struct TRI_Header header;
   float * triangleVertex;
   float * normal;
};



#if HAVE_OBJ_CODE_AVAILIABLE
int convertObjToTri(struct TRI_Model * tri , struct OBJ_Model * obj);
#endif // HAVE_OBJ_CODE_AVAILIABLE

int loadModelTri(const char * filename , struct TRI_Model * triModel);
int saveModelTri(const char * filename , struct TRI_Model * triModel);
int saveModelTriHeader(const char * filename , struct TRI_Model * triModel);



#endif // MODEL_LOADER_TRI_H_INCLUDED
