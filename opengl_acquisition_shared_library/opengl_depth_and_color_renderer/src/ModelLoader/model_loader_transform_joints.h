/** @file model_loader_transform_joints.h
 *  @brief  TRIModels loader/writer and basic functions
            part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer

            The purpose of this code is to perform all the necessary transformations on TRI Models based on their bones and poses we want to achieve
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED
#define MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED


#include "model_loader_tri.h"

#ifdef __cplusplus
extern "C"
{
#endif


#define MAX_BONES_PER_VERTICE 4
struct TRI_Bones_Per_Vertex_Vertice_Item
{
  unsigned int bonesOfthisVertex;
  float weightsOfThisVertex[MAX_BONES_PER_VERTICE];
  unsigned int indicesOfThisVertex[MAX_BONES_PER_VERTICE];
  unsigned int boneIDOfThisVertex[MAX_BONES_PER_VERTICE];
};


struct TRI_Bones_Per_Vertex
{
  unsigned int numberOfBones;
  unsigned int numberOfVertices;
  unsigned int maxBonesPerVertex;

  struct TRI_Bones_Per_Vertex_Vertice_Item * bonesPerVertex;
};



struct TRI_Bones_Per_Vertex * allocTransformTRIBonesToVertexBoneFormat(struct TRI_Model * in);
void freeTransformTRIBonesToVertexBoneFormat(struct TRI_Bones_Per_Vertex * in);

/**
* @brief Alter color information of model to reflect bone IDs
* @ingroup TRI
* @param  input TRI structure that we are going to work on..!
*/
void colorCodeBones(struct TRI_Model * in);

/**
* @brief Transform a TRI Joint using just 3 Euler Angles..!
* @ingroup TRI
* @param  input TRI structure with the loaded model
* @param  allocated matrix array that will be altered
* @param  size of allocated matrix array

* @param  The joint to select in->bones[jointToChange].boneName to see what it was
* @param  Rotation in Euler Angle axis X
* @param  Rotation in Euler Angle axis Y
* @param  Rotation in Euler Angle axis Z
*/
void transformTRIJoint(
                        struct TRI_Model * in ,
                        float * jointData ,
                        unsigned int jointDataSize ,

                        unsigned int jointToChange ,
                        float rotEulerX ,
                        float rotEulerY ,
                        float rotEulerZ
                      );



/**
* @brief Allocate all the 4x4 matrices needed to control a TRI_Model , this call also initializes them so they are ready for use..! , they need to be freed after being used
* @ingroup TRI
* @param  input TRI structure with the loaded model we want to allocate matrices for
* @param  output number of 4x4 matrices allocated
* @retval 0=Failure or else a pointer to an array of 4x4 matrices
*/
float * mallocModelTransformJoints(
                                    struct TRI_Model * triModelInput ,
                                    unsigned int * jointDataSizeOutput
                                  );


/**
* @brief  Do model transform based on joints
* @ingroup TRI
* @param  output TRI structure transformed as requested
* @param  input TRI structure
* @param  array of joints allocated using mallocModelTransformJoints
* @param  number of joints got from the jointDataSizeOutput of mallocModelTransformJoints
* @param  autodetect non-identity matrices on jointdata array and skip some calculations
* @param  perform vertex transforms ( if not triModelOut will be the same as triModelIn )
* @retval 0=Failure1=Success
*/
int doModelTransform(
                      struct TRI_Model * triModelOut ,
                      struct TRI_Model * triModelIn ,
                      float * jointData ,
                      unsigned int jointDataSize ,
                      unsigned int autodetectAlteredMatrices,
                      unsigned int performVertexTransform
                    );


#ifdef __cplusplus
}
#endif

#endif // MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED
