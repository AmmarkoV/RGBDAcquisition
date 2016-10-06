/** @file model_loader_tri.h
 *  @brief  TRIModels loader/writer and basic functions
            part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
            The basic idea is a 3D mesh container that
            1) Contains Vertices/Colors/TextureCoords/Normals  indexed or not
            2) Contains skinning information
            3) Is a binary format , so ( smaller filesizes )
            4) Does not have dependencies ( plain C code )
            5) Is as small and clear as possible
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MODEL_LOADER_TRI_H_INCLUDED
#define MODEL_LOADER_TRI_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

#include "model_loader_setup.h"

#define TRI_LOADER_VERSION 5

struct TRI_Header
{
     char TRIMagic[5];
     unsigned int triType;
     unsigned int floatSize;
     //These first values guarantee that there is compatibility between machines/versions etc
     unsigned int drawType; //0 = triangles , 1 = quads , // etc ( not used yet )

     unsigned int numberOfVertices; // the number of vertices
     unsigned int numberOfNormals;
     unsigned int numberOfTextureCoords;
     unsigned int numberOfColors;
     unsigned int numberOfIndices;
     unsigned int numberOfBones;
     unsigned int rootBone;
     double boneGlobalInverseTransform[16];

     //In order not to break this file format ever again
     unsigned int notUsed1;
     unsigned int notUsed2;
     unsigned int notUsed3;
     unsigned int notUsed4;
     unsigned int notUsed5;
};

struct TRI_Bones_Header
{
  unsigned int boneParent;
//-------------------------------------------
  unsigned int boneWeightsNumber;
  unsigned int boneNameSize;
//-------------------------------------------
  double matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose[16]; //selfdescriptive
  double finalVertexTransformation[16]; //What we will use in the end
  double localTransformation[16]; // or node->mTransformation
  unsigned char altered;

  double minXRotation , x , maxXRotation;
  double minYRotation , y , maxYRotation;
  double minZRotation , z , maxZRotation;
  unsigned char rotationLimitsSet;


//-------------------------------------------
  unsigned int  allocatedNumberOfBoneChildren; //This is used when doing recursions , should be the same with numberOfBoneChildren
  unsigned int  numberOfBoneChildren;
//-------------------------------------------
};

struct TRI_Bones
{
  struct TRI_Bones_Header * info;
  char*  boneName;
  float * weightValue;
  unsigned int * weightIndex;
  unsigned int * boneChild;  //bone child structure 0-numberOfBoneChildren of bone ids
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



/**
* @brief Printout the values of an array of 16 values ( double val[16]; ) that contains a 4x4 Matrix
* @ingroup TRI
* @param  string label to include in the printout
* @param  pointer to the 4x4 double matrix
*/
void print4x4DMatrixTRI(char * str , double * matrix4x4);

/**
* @brief Printout the bone structure of a TRI model
* @ingroup TRI
* @param  input TRI structure with the loaded model we want a printout of
* @param  switch to control printout of matrices
*/
void printTRIBoneStructure(struct TRI_Model * triModel,int alsoPrintMatrices);

int fillFlatModelTriFromIndexedModelTri(struct TRI_Model * triModel , struct TRI_Model * indexed);



/**
* @brief Allocate the space for a TRI model ( possibly to build a model from scratch ) , typically you don't want to do this , just use loadModelTri instead..!
* @ingroup TRI
* @param  input TRI structure with the loaded model we want freed
*/
struct TRI_Model * allocateModelTri();

/**
* @brief After being done with the model we can deallocate it , the model needs to have been allocated with allocateModelTri to be correctly freed
* @ingroup TRI
* @param  input TRI structure with the loaded model we want freed
*/
int freeModelTri(struct TRI_Model * triModel);

/**
* @brief After being done with the model we can deallocate its internal structures , this doesn't do the final free(triModel) call..! , see freeModelTri call for a call that does this plus the final free call
* @ingroup TRI
* @param  input TRI structure with the loaded model we want its internals freed
*/
void deallocInternalsOfModelTri(struct TRI_Model * triModel);


int loadModelTri(const char * filename , struct TRI_Model * triModel);
int saveModelTri(const char * filename , struct TRI_Model * triModel);

int findTRIBoneWithName(struct TRI_Model * triModel ,const char * name , unsigned int * boneNumResult);

void copyModelTriHeader(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN );
void copyModelTri(struct TRI_Model * triModelOUT , struct TRI_Model * triModelIN , int copyBoneStructures);


/**
* @brief If INCLUDE_OPENGL_CODE is declared ( so we have an openGL context )  we can use the fixed graphics pipeline and do the rendering of the file on the spot.
* @ingroup TRI
* @param  input TRI structure with the loaded model we want to render
*/
void doTriDrawCalllist(struct TRI_Model * tri );

#ifdef __cplusplus
}
#endif

#endif // MODEL_LOADER_TRI_H_INCLUDED
