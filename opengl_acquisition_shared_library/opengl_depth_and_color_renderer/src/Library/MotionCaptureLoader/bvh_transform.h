#ifndef BVH_TRANSFORM_H_INCLUDED
#define BVH_TRANSFORM_H_INCLUDED

#include "bvh_loader.h"

/*
   0  1  2  3
   4  5  6  7
   8  9 10 11
  12 13 14 15
*/

struct BVH_TransformedJoint
{
  double finalVertexTransformation[16]; //What we will use in the end
  double localTransformation[16]; // or node->mTransformation
};


struct BVH_Transform
{
  struct BVH_TransformedJoint joint[MAX_BVH_JOINT_HIERARCHY_SIZE];
};



int bvh_loadTransformForFrame(
                               struct BVH_MotionCapture * bvhMotion ,
                               BVHFrameID fID ,
                               struct BVH_Transform * bvhTransform
                             );

#endif // BVH_TRANSFORM_H_INCLUDED
