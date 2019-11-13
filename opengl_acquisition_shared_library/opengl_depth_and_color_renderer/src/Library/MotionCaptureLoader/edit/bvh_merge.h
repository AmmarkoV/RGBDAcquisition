#ifndef BVH_MERGE_H_INCLUDED
#define BVH_MERGE_H_INCLUDED



#include "../bvh_loader.h"

int bvh_mergeWith(
                   struct BVH_MotionCapture * mc,
                   struct BVH_MotionCapture * mcToMerge,
                   const char * pathToMergeRules
                 );



#endif
