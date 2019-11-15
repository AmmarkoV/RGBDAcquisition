#ifndef BVH_MERGE_H_INCLUDED
#define BVH_MERGE_H_INCLUDED



#include "../bvh_loader.h"


struct BVH_MergeAssociations
{
   unsigned int jointAssociationTargetToSource[MAX_BVH_JOINT_HIERARCHY_SIZE];
   unsigned int jointAssociationSourceToTarget[MAX_BVH_JOINT_HIERARCHY_SIZE];


};


int bvh_mergeWith(
                   struct BVH_MotionCapture * mc,
                   struct BVH_MotionCapture * mcToMerge,
                   const char * pathToMergeRules
                 );



#endif
