#include "bvh_merge.h"
#include "../bvh_to_tri_pose.h"
#include "bvh_cut_paste.h"


#include <stdio.h>
#include <stdlib.h>

// ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --merge Motions/49_04.bvh Motions/mergeDazFriendlyMoreThan19.profile --bvh testMerge.bvh
// ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --print --merge Motions/02_03.bvh Motions/mergeDazFriendlyAndAddHead.profile --bvh testMerge.bvh


int bvh_mergeWith(
                   struct BVH_MotionCapture * targetMC,
                   struct BVH_MotionCapture * sourceMC,
                   const char * pathToMergeRules
                 )
{
 fprintf(stderr,"bvh_mergeWith will try to merge the skeleton and motion frames of file %s in target file (%s) \n",sourceMC->fileName,targetMC->fileName);
 struct bvhToTRI associationRuleLabels={0};
 if (bvh_loadBVHToTRIAssociationFile(pathToMergeRules,&associationRuleLabels,targetMC))
 {
   if (
        bvh_EraseAndAllocateSpaceForNumberOfFrames(
                                                   targetMC,
                                                   sourceMC->numberOfFramesEncountered
                                                  )
      )
   {
     //Our target file has enough space to accomodate the incoming motion buffers..!
     if (sourceMC->numberOfFrames!=targetMC->numberOfFrames)
     {
      fprintf(stderr,"CodeBug: we don't have enough space to write\n");
     }

     struct BVH_MergeAssociations rules;

     unsigned int sourceJID=0,targetJID=0;
     if ( (sourceMC->jointHierarchySize>=MAX_BVH_JOINT_HIERARCHY_SIZE) || (targetMC->jointHierarchySize>=MAX_BVH_JOINT_HIERARCHY_SIZE) )
     {
     for (sourceJID=0; sourceJID<sourceMC->jointHierarchySize; sourceJID++)
     {
      for (targetJID=0; targetJID<targetMC->jointHierarchySize; targetJID++)
      {
        if (sourceMC->jointHierarchy[sourceJID].jointNameHash == targetMC->jointHierarchy[targetJID].jointNameHash )
        {
         rules.jointAssociationSourceToTarget[sourceJID]=targetJID;
         rules.jointAssociationTargetToSource[targetJID]=sourceJID;
        }
       }
      }
     } else
     {
       fprintf(stderr,"Joint hierarchy is too big..\n");
     }

     unsigned int frameID=0,sourceMID=0;
     for (frameID=0; frameID<sourceMC->numberOfFrames; frameID++)
     {
       for (sourceMID=0; sourceMID<sourceMC->numberOfValuesPerFrame; sourceMID++)
       {



       }
     }

     return 0;
   } else
   { fprintf(stderr,"Could not allocate enough space to hold the merged motion buffers..\n"); }
 } else
 { fprintf(stderr,"Could not parse rule file (%s) that contains the merge mapping\n",pathToMergeRules); }

 return 0;
}
