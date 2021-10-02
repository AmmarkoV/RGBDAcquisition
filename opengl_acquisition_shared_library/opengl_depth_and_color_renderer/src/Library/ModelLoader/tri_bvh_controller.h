/** @file tri_bvh_controller.h
 *  @brief  A module that can transform TRI models from BVH files
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef TRI_BVH_CONTROLLER_H_INCLUDED
#define TRI_BVH_CONTROLLER_H_INCLUDED

#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"
#include "../MotionCaptureLoader/bvh_loader.h"
#include "../MotionCaptureLoader/calculate/bvh_transform.h"

const int animateTRIModelUsingBVHArmature(struct TRI_Model * model,struct BVH_MotionCapture * bvh,unsigned int frameID)
{
 if (model==0) { return 0; }
 if (bvh==0)   { return 0; }
 //--------------------------

  struct BVH_Transform bvhTransform={0};
  if (
       bvh_loadTransformForFrame(
                                  bvh,
                                  frameID,
                                  &bvhTransform,
                                  0
                                )
     )
     {



     }

 return 0;
}




#endif //TRI_BVH_CONTROLLER_H_INCLUDED
