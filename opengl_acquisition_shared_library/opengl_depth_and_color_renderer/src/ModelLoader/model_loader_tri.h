#ifndef MODEL_LOADER_TRI_H_INCLUDED
#define MODEL_LOADER_TRI_H_INCLUDED



struct TRI_Model
{
   unsigned int triType;

   unsigned int numberOfTriangles;
   float * triangleVertex;

   unsigned int numberOfNormals;
   float * normal;
};



int loadModelTri(const char * filename , struct TRI_Model * triModel);
int saveModelTri(const char * filename , struct TRI_Model * triModel);
int saveModelTriHeader(const char * filename , struct TRI_Model * triModel);



#endif // MODEL_LOADER_TRI_H_INCLUDED
