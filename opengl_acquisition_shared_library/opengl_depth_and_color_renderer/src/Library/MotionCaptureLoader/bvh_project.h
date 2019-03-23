#ifndef BVH_PROJECT_H_INCLUDED
#define BVH_PROJECT_H_INCLUDED

#include "bvh_loader.h"
#include "bvh_transform.h"


#include "../../../../../tools/AmMatrix/simpleRenderer.h"



struct BVH_RendererConfiguration
{
  //Intrinsics
  float fX,fY,cX,cY;
  //Distortion
  float k1,k2,k3,p1,p2;
  //----------
  float View[16];
  float R[9];
  float T[3];
  float projection[16];
  float viewport[4];
};



void bvh_cleanTransform(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform     * bvhTransform
                      );

int bvh_projectTo2D(
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform     * bvhTransform,
                     struct simpleRenderer    * renderer,
                     unsigned int               occlusions
                   );

#endif // BVH_PROJECT_H_INCLUDED
