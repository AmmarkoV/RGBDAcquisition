#ifndef MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED
#define MODEL_LOADER_TRANSFORM_JOINTS_H_INCLUDED


#include "model_loader_tri.h"

#ifdef __cplusplus
extern "C"
{
#endif

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
