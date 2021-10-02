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





/**
* @brief TRI_LOADER_VERSION is a compatibility switch , every time this changes it invalidates all older files in order
         to keep the spec as clean as possible
* @ingroup TRI
*/
#define TRI_LOADER_VERSION 10
///IF I EVER CHANGE THE VERSION AGAIN I SHOULD ALWAYS UPDATE LAST STABLE COMMIT INSIDE MODEL_LOADER_TRI..!








/**
* @brief Each bone has a parent ( if the parent has the same ID as the node it is the root )  , some transforms , weights and limits
         they are all stored here
* @ingroup TRI
*/
struct TRI_Bones_Header
{
  unsigned int boneParent;
//-------------------------------------------
  unsigned int boneWeightsNumber;
  unsigned int boneNameSize;
//-------------------------------------------
  float matrixThatTransformsFromMeshSpaceToBoneSpaceInBindPose[16]; //selfdescriptive
  float finalVertexTransformation[16]; //What we will use in the end
  float localTransformation[16]; // or node->mTransformation
  unsigned char altered;

  //Bone center location and dimension
  float x , dimX;
  float y , dimY;
  float z , dimZ;
  unsigned char bonePositionSet;

  //Bone rotation setting
  float minXRotation , rotX , maxXRotation;
  float minYRotation , rotY , maxYRotation;
  float minZRotation , rotZ , maxZRotation;
  float minWRotation , rotW , maxWRotation;

  unsigned char rotationSet , rotationLimitsSet;
  unsigned char isEulerRotation;
  unsigned char eulerRotationOrder;
  unsigned char isQuaternionRotation;
//-------------------------------------------
  unsigned int  allocatedNumberOfBoneChildren; //This is used when doing recursions , should be the same with numberOfBoneChildren
  unsigned int  numberOfBoneChildren;
//-------------------------------------------


 //In order not to break this file format ever again
 unsigned char notUsed1;
 unsigned char notUsed2;
 unsigned int notUsed3;
 unsigned int notUsed4;
};

/**
* @brief Each bone has some static attributes stored in the info header and a variable number of weights/childs etc
* @ingroup TRI
*/
struct TRI_Bones
{
  struct TRI_Bones_Header * info;
  char *  boneName;
  float * weightValue;
  unsigned int * weightIndex;
  unsigned int * boneChild;  //bone child structure 0-numberOfBoneChildren of bone ids
};



/**
* @brief The header and initial file block of the TRI format , magic for the files is TRI3D
* @ingroup TRI
*/
struct TRI_Header
{
     char TRIMagic[5]; // TRI3D
     unsigned int triType;
     unsigned int nameSize;
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
     float boneGlobalInverseTransform[16];

     //In order not to break this file format ever again
     unsigned int notUsed1;
     unsigned int notUsed2;
     unsigned int notUsed3;
     unsigned int notUsed4;
     unsigned int notUsed5;
};


/**
* @brief A TRI Model skeleton in all its simplicity..!
* @ingroup TRI
*/
struct TRI_Model
{
   struct TRI_Header header;
   char *  name;
   float * vertices;
   float * normal;
   float * textureCoords;
   float * colors;
   struct TRI_Bones * bones;
   unsigned int * indices;
};




struct TRI_Container_Header
{
     char TRIMagic[5]; // TRI3D
     unsigned int triType;
     unsigned int nameSize;
     unsigned int floatSize;
     //These first values guarantee that there is compatibility between machines/versions etc

     unsigned int numberOfMeshes; // the number of vertices

     // - - - - - - - - - -
     unsigned int notUsed1;
     unsigned int notUsed2;
     unsigned int notUsed3;
     unsigned int notUsed4;
     unsigned int notUsed5;
     // - - - - - - - - - -
};



/**
* @brief A TRI Container that might contain multiple meshes..!
* @ingroup TRI
*/
struct TRI_Container
{
   struct TRI_Container_Header header;
   char *  name;

   float * meshTransformation;
   struct TRI_Model * mesh;
};




/**
* @brief Printout the values of an array of 16 values ( float val[16]; ) that contains a 4x4 Matrix
* @ingroup TRI
* @param  string label to include in the printout
* @param  pointer to the 4x4 float matrix
*/
void print4x4FMatrixTRI(const char * str , float * matrix4x4);

/**
* @brief Printout the bone structure of a TRI model
* @ingroup TRI
* @param  input TRI structure with the loaded model we want a printout of
* @param  switch to control printout of matrices
*/
void printTRIBoneStructure(struct TRI_Model * triModel,int alsoPrintMatrices);

/**
* @brief This function can flatten out an indexed TRI_Model , so it becomes literally an array of triangles with no indexing.
*        Of course this will result in a bigger chunk of memory required , but it might be useful
* @ingroup TRI
* @param  output TRI structure with the resulting flat model , should be allocated via allocateModelTri
* @param  input TRI structure with the loaded index model we want to process
* @retval 0=Failure,1=Success
*/
int fillFlatModelTriFromIndexedModelTri(struct TRI_Model * triModel , struct TRI_Model * indexed);

/**
* @brief Allocate the space for a TRI model ( possibly to build a model from scratch ) , typically you don't want to do this , just use loadModelTri instead..!
* @ingroup TRI
* @param  input TRI structure with the loaded model we want freed
* @retval 0=Failure or else a pointer to a newly allocated TRI_Model
*/
struct TRI_Model * allocateModelTri();

/**
* @brief After being done with the model we can deallocate it , the model needs to have been allocated with allocateModelTri to be correctly freed
* @ingroup TRI
* @param  input TRI structure with the loaded model we want freed
* @retval 0=Failure,1=Success
*/
int freeModelTri(struct TRI_Model * triModel);

/**
* @brief After being done with the model we can deallocate its internal structures , this doesn't do the final free(triModel) call..! , see freeModelTri call for a call that does this plus the final free call
* @ingroup TRI
* @param  input TRI structure with the loaded model we want its internals freed
*/
void deallocInternalsOfModelTri(struct TRI_Model * triModel);


/**
* @brief  Load TRI model from a file
* @ingroup TRI
* @param  String with the filename we want to load from
* @param  output structure to hold the TRI Model ( allocate with allocateModelTri )
* @retval 0=Failure,1=Success
*/
int loadModelTri(const char * filename , struct TRI_Model * triModel);



/**
* @brief  Force colors on vertices by sampling a texture
* @ingroup TRI
* @param  input TRI structure with mesh we want to color
* @param  Pixel images
* @param  Width of image
* @param  Height of image
* @param  Bitsperpixel
* @param  channels of image
* @retval 0=Failure,1=Success
*/
int paintTRIUsingTexture(struct TRI_Model * triModel,unsigned char * pixels , unsigned int width ,unsigned int height, unsigned int bitsperpixel , unsigned int channels);


/**
* @brief  Force a specific color on all vertices of TRI file
* @ingroup TRI
* @param  input TRI structure with mesh we want to color
* @param  R channel value ( 0-255 )
* @param  G channel value ( 0-255 )
* @param  B channel value ( 0-255 )
* @retval 0=Failure,1=Success
*/
int paintTRI(struct TRI_Model * triModel,char r, char g, char b);

/**
* @brief  Save TRI model to a file
* @ingroup TRI
* @param  String with the filename we want to load from
* @param  input structure that hold the TRI Model we want to save
* @retval 0=Failure,1=Success
*/
int saveModelTri(const char * filename , struct TRI_Model * triModel);




/**
* @brief  Convert all TRI Bone names to lowercase to make them easier to parse
* @ingroup TRI
* @param  TRI structure with the bone names we want to make lowercase
* @retval 0=Failure,1=Success
*/
int makeAllTRIBoneNamesLowerCase(struct TRI_Model * triModel);


/**
* @brief  Search inside the bone tree of a TRI Model and get back a specific boneID
* @ingroup TRI
* @param  input TRI structure with the bones we want to search
* @param  string with the name of the bone we are looking for
* @param  output boneID if we found one
* @retval 0=Bone Does not exist , 1=Bone found
*/
int findTRIBoneWithName(struct TRI_Model * triModel ,const char * searchName , unsigned int * boneIDResult);

/**
* @brief One pretty standard operations that is needed often is copying models around to edit them without destroying the original
         This function does exactly that , with or without the bone structure.
* @ingroup TRI
* @param  output TRI model that will contain our fresh deep copy (  allocate with allocateModelTri )
* @param  input TRI model
* @param  switch to control copying bones
*/
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

