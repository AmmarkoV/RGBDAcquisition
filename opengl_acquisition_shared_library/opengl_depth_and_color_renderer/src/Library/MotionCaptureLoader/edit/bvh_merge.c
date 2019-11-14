#include "bvh_merge.h"
#include "../bvh_to_tri_pose.h"

// ./BVHTester --from Motions/02_03.bvh --merge Motions/49_04.bvh Motions/mergeDazFriendlyMoreThan19.profile --bvh testMerge.bvh

int bvh_mergeWith(
                   struct BVH_MotionCapture * mc,
                   struct BVH_MotionCapture * mcToMerge,
                   const char * pathToMergeRules
                 )
{
 struct bvhToTRI bvhtri={0};
 bvh_loadBVHToTRIAssociationFile(pathToMergeRules,&bvhtri,mc);

 return 0;
}
