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
* @brief Allocate all the 4x4 matrices needed to control a TRI_Model , this call also initializes them so they are ready for use..! , they need to be freed after beeing used
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
* @retval 0=Failure1=Success
*/
int doModelTransform(
                      struct TRI_Model * triModelOut ,
                      struct TRI_Model * triModelIn ,
                      float * jointData ,
                      unsigned int jointDataSize
                    );


#ifdef __cplusplus
}
#endif

#endif // MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED
