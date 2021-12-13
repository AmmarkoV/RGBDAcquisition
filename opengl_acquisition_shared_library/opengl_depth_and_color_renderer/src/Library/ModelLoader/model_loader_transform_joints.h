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


/**
* @brief This is the maximum number of bones per vertice this is needed to allocate correctly the arrays on TRI_Bones_Per_Vertex_Vertice_Item , 4 is
         a logical value..
* @ingroup TRI
*/
#define MAX_BONES_PER_VERTICE 4
struct TRI_Bones_Per_Vertex_Vertice_Item
{
  unsigned int bonesOfthisVertex;
  float weightsOfThisVertex[MAX_BONES_PER_VERTICE];
  unsigned int indicesOfThisVertex[MAX_BONES_PER_VERTICE];
  unsigned int boneIDOfThisVertex[MAX_BONES_PER_VERTICE];
};


/**
* @brief A different way to store TRI bones , per vertex ( for shader pose configuration )
* @ingroup TRI
*/
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
* @brief Get the palette that colorCodeBones will use to reflect bone IDs
* @ingroup TRI
* @param  input TRI structure that we are going to work on..!
*/
float * generatePalette(struct TRI_Model * in);





unsigned int * convertTRIBonesToParentList(struct TRI_Model * in , unsigned int * outputNumberOfBones);

/**
* @brief Populate in->bone[x].info->x/y/z with the centers of the bone based on the current setup of the
         model..
* @ingroup TRI
* @param  input TRI structure that we are going to work on..!
* @param  output number of bones allocated ..!
* @retval 0=Failure or else a pointer to an array of float triplets with x/y/z locations of each bone.
* @bug  Please note that prior calling this function the program should have done a applyVertexTransformation or doModelTransform to set up the vertices , because this function just calculates
the average of each bone. Please note that the output is in the coordinate space of the binding pose model and needs to be transformed/projected etc
according to the real location of the mesh , this function is also quite resource heavy and needs to be improved..
*/
float * convertTRIBonesToJointPositions(struct TRI_Model * in , unsigned int * outputNumberOfJoints);


unsigned int  * getClosestVertexToJointPosition(struct TRI_Model * in , float * joints , unsigned int numberOfJoints);


int tri_colorCodeTexture(struct TRI_Model * in, unsigned int x, unsigned int y, unsigned int width,unsigned int height);



int setTRIJointRotationOrder(
                              struct TRI_Model * in ,
                              unsigned int jointToChange ,
                              unsigned int rotationOrder
                             );

int getTRIJointRotationOrder(
                             struct TRI_Model * in ,
                             unsigned int jointToChange ,
                             unsigned int rotationOrder
                            );




/**
* @brief Populate in->bone[x].info->x/y/z with the centers of the bone based on the current setup of the
         model..
* @ingroup TRI
* @param  input TRI structure that we are going to work on..!
*/
int setTRIModelBoneInitialPosition(struct TRI_Model * in);



/**
* @brief Alter color information of model to reflect bone IDs
* @ingroup TRI
* @param  input TRI structure that we are going to work on..!
*/
void colorCodeBones(struct TRI_Model * in);





/**
* @brief Transform a TRI Joint using just 3 Euler Angles using ZYX order..!
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



float * mallocModelTransformJointsEulerAnglesDegrees(
                                                      struct TRI_Model * triModelInput ,
                                                      float * jointData ,
                                                      unsigned int jointDataSize ,
                                                      unsigned int method
                                                     );


int applyVertexTransformation( struct TRI_Model * triModelOut , struct TRI_Model * triModelIn );

void printModelTransform(struct TRI_Model * in);

/**
* @brief  Do model transform based on joints
* @ingroup TRI
* @param  output TRI structure transformed as requested
* @param  input TRI structure
* @param  array of joints allocated using mallocModelTransformJoints
* @param  number of joints got from the jointDataSizeOutput of mallocModelTransformJoints
* @param  autodetect non-identity matrices on jointdata array and skip some calculations
* @param  use direct setting of final matrices ( only needed if you really know what you want in the final matrices )
* @param  perform vertex transforms ( if not triModelOut will be the same as triModelIn )
* @param  jointAxisConvention , 0 = default
* @retval 0=Failure,1=Success
*/
int doModelTransform(
                      struct TRI_Model * triModelOut ,
                      struct TRI_Model * triModelIn ,
                      float * joint4x4Data ,
                      unsigned int joint4x4DataSize ,
                      unsigned int autodetectAlteredMatrices,
                      unsigned int directSettingOfMatrices ,
                      unsigned int performVertexTransform  ,
                      unsigned int jointAxisConvention
                    );


#ifdef __cplusplus
}
#endif

#endif // MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED
