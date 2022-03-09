#ifndef BVH_PROJECT_H_INCLUDED
#define BVH_PROJECT_H_INCLUDED

#include "../bvh_loader.h"
#include "bvh_transform.h"
#include "../mathLibrary.h"

#ifdef __cplusplus
extern "C"
{
#endif


struct BVH_RendererConfiguration
{
  int isDefined;
  unsigned int width;
  unsigned int height;

  //Frustrum limits
  float near,far;

  //Intrinsics
  float fX,fY,cX,cY;
  //Distortion
  float k1,k2,k3,p1,p2;
  //----------
  //float R[9];
  float T[4];
  struct Matrix4x4OfFloats projection;
  struct Matrix4x4OfFloats viewMatrix;
  int viewport[4];
};



void bvh_cleanTransform(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform     * bvhTransform
                      );


int bvh_projectJIDTo2D(
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform     * bvhTransform,
                     struct simpleRenderer    * renderer,
                     BVHJointID jID,
                     unsigned int               occlusions,
                     unsigned int               directRendering
                   );

int bvh_projectTo2D(
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform     * bvhTransform,
                     struct simpleRenderer    * renderer,
                     unsigned int               occlusions,
                     unsigned int               directRendering
                   );


#ifdef __cplusplus
}
#endif



#endif // BVH_PROJECT_H_INCLUDED
