#ifndef BVH_TO_TRI_POSE_H_INCLUDED
#define BVH_TO_TRI_POSE_H_INCLUDED

#include "bvh_loader.h"

struct BVH_JointAssocation
{
  const char * bvhJointName[MAX_BVH_JOINT_NAME];
  const char * triJointName[MAX_BVH_JOINT_NAME];
};

struct bvhToTRI
{
 struct BVH_JointAssocation * jointAssociation[MAX_BVH_JOINT_HIERARCHY_SIZE];
};

#endif // BVH_TO_TRI_POSE_H_INCLUDED
