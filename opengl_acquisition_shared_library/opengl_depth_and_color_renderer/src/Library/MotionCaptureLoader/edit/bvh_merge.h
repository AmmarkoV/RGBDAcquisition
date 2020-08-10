#ifndef BVH_MERGE_H_INCLUDED
#define BVH_MERGE_H_INCLUDED



#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

struct BVH_MergeAssociations
{
   char jointAssociationSourceToTargetExists[MAX_BVH_JOINT_HIERARCHY_SIZE];
   char jointAssociationTargetToSourceExists[MAX_BVH_JOINT_HIERARCHY_SIZE];
   unsigned int jointAssociationTargetToSource[MAX_BVH_JOINT_HIERARCHY_SIZE];
   unsigned int jointAssociationSourceToTarget[MAX_BVH_JOINT_HIERARCHY_SIZE];


};

int bvh_mergeWith(
                   struct BVH_MotionCapture * targetMC,
                   struct BVH_MotionCapture * sourceMC,
                   const char * pathToMergeRules
                 );

#ifdef __cplusplus
}
#endif

#endif
