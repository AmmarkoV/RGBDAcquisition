#ifndef BVH_MERGE_H_INCLUDED
#define BVH_MERGE_H_INCLUDED



#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

struct BVH_MergeAssociations
{
   char * jointAssociationSourceToTargetExists;
   char * jointAssociationTargetToSourceExists;
   unsigned int * jointAssociationTargetToSource;
   unsigned int * jointAssociationSourceToTarget;
};

int bvh_mergeWith(
                   struct BVH_MotionCapture * targetMC,
                   struct BVH_MotionCapture * sourceMC,
                   const char * pathToMergeRules
                 );



int bvh_mergeOffsetsInMotions(
                               struct BVH_MotionCapture * mc
                             );


int bvh_mergeFacesRobot(
                         int startAt,
                         int argc,
                         const char **argv
                        );

#ifdef __cplusplus
}
#endif

#endif
