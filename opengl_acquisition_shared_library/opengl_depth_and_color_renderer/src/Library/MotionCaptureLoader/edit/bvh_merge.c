#include "bvh_merge.h"
#include "../calculate/bvh_to_tri_pose.h"
#include "bvh_cut_paste.h"

#include <stdio.h>
#include <stdlib.h>

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

// ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --merge Motions/49_04.bvh Motions/mergeDazFriendlyMoreThan19.profile --bvh testMerge.bvh
// ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --print --merge Motions/02_03.bvh Motions/mergeDazFriendlyAndAddHead.profile --bvh testMerge.bvh
// ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --print --merge Motions/102_01cc.bvh Motions/mergeDazFriendlyAndAddHead.profile --bvh testMerge2.bvh

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

     struct BVH_MergeAssociations rules={0};
 
     if ( (sourceMC->jointHierarchySize<MAX_BVH_JOINT_HIERARCHY_SIZE) && (targetMC->jointHierarchySize<MAX_BVH_JOINT_HIERARCHY_SIZE) )
     {
      for (unsigned int sourceJID=0; sourceJID<sourceMC->jointHierarchySize; sourceJID++)
      {
      for (unsigned int targetJID=0; targetJID<targetMC->jointHierarchySize; targetJID++)
      {
       if (!sourceMC->jointHierarchy[sourceJID].isEndSite)
       {
        if (sourceMC->jointHierarchy[sourceJID].jointNameHash == targetMC->jointHierarchy[targetJID].jointNameHash )
        {
         rules.jointAssociationSourceToTargetExists[sourceJID]=1;
         rules.jointAssociationSourceToTarget[sourceJID]=targetJID;
         rules.jointAssociationTargetToSourceExists[targetJID]=1;
         rules.jointAssociationTargetToSource[targetJID]=sourceJID;
         fprintf(
                 stderr,GREEN "Correspondace of source(%s/%u)/target(%s/%u) ..\n" NORMAL,
                 sourceMC->jointHierarchy[sourceJID].jointName,sourceJID,
                 targetMC->jointHierarchy[targetJID].jointName,targetJID
                );
        }
       }
      }
      }
     } else
     {
       fprintf(stderr,RED "Joint hierarchy is too big..\n" NORMAL);
       fprintf(stderr,"Source Joint Hierarchy has a size of %u ( max is %d )\n",sourceMC->jointHierarchySize,MAX_BVH_JOINT_HIERARCHY_SIZE);
       fprintf(stderr,"Target Joint Hierarchy has a size of %u ( max is %d )\n",targetMC->jointHierarchySize,MAX_BVH_JOINT_HIERARCHY_SIZE);
     }

     fprintf(stderr,"Will copy %u frames from source to target",targetMC->numberOfFrames);

     for (unsigned int frameID=0; frameID<targetMC->numberOfFrames; frameID++)
     {
       for (unsigned int sourceMID=0; sourceMID<sourceMC->numberOfValuesPerFrame; sourceMID++)
       {
         unsigned int sourceJID=sourceMC->motionToJointLookup[sourceMID].jointID;// channelIDMotionOffset
         unsigned int channel=sourceMC->motionToJointLookup[sourceMID].channelID;
         if (rules.jointAssociationSourceToTargetExists[sourceJID])
         {
             unsigned int targetJID=rules.jointAssociationSourceToTarget[sourceJID];
             //targetMID=targetMC->jointToMotionLookup[targetJID].jointMotionOffset+channel;
             unsigned int targetMID=targetMC->jointToMotionLookup[targetJID].channelIDMotionOffset[channel];
             targetMC->motionValues[(frameID)*targetMC->numberOfValuesPerFrame + targetMID] = sourceMC->motionValues[(frameID)*sourceMC->numberOfValuesPerFrame + sourceMID];
         }
       }
     }

     targetMC->numberOfFramesEncountered = sourceMC->numberOfFramesEncountered;

     return 1;
   } else
   { fprintf(stderr,RED "Could not allocate enough space to hold the merged motion buffers..\n" NORMAL); }
 } else
 { fprintf(stderr,RED "Could not parse rule file (%s) that contains the merge mapping\n" NORMAL,pathToMergeRules); }

 return 0;
}
