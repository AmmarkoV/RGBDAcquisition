#ifndef BVH_TRANSFORM_H_INCLUDED
#define BVH_TRANSFORM_H_INCLUDED

#include "bvh_loader.h"

struct BVH_TransformedJoint
{
  double finalVertexTransformation[16]; //What we will use in the end
  double localTransformation[16]; // or node->mTransformation
};


struct BVH_Transform
{
  struct BVH_TransformedJoint joint[MAX_BVH_JOINT_HIERARCHY_SIZE];
};
#endif // BVH_TRANSFORM_H_INCLUDED
