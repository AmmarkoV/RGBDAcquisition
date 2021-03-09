#include "bvh_merge.h"
#include "../calculate/bvh_to_tri_pose.h"
#include "bvh_cut_paste.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

// ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --merge Motions/49_04.bvh Motions/mergeDazFriendlyMoreThan19.profile --bvh testMerge.bvh
// ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --print --merge Motions/02_03.bvh Motions/mergeDazFriendlyAndAddHead.profile --bvh testMerge.bvh
// ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --print --merge Motions/102_01cc.bvh Motions/mergeDazFriendlyAndAddHead.profile --bvh testMerge2.bvh
// ./BVHTester --from Motions/head.bvh --merge Motions/heads/face_65_eyesupright.bvh Motions/mergeHead.profile --bvh test.bvh 
// ./BVHTester --from Motions/headerWithHeadAndOneMotion.bvh --merge Motions/dance2_subject2.bvh Motions/mergeLafanWithMakehuman.profile --bvh test.bvh 

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
     
     unsigned int maxOfMaxJointHierarchySize = sourceMC->MAX_jointHierarchySize;
     if (maxOfMaxJointHierarchySize<targetMC->MAX_jointHierarchySize) {  maxOfMaxJointHierarchySize=targetMC->MAX_jointHierarchySize; }
     
     rules.jointAssociationSourceToTargetExists = (char*)    malloc(sizeof(char) * maxOfMaxJointHierarchySize);
     rules.jointAssociationTargetToSourceExists = (char*)    malloc(sizeof(char) * maxOfMaxJointHierarchySize);
     rules.jointAssociationTargetToSource = (unsigned int*)  malloc(sizeof(unsigned int) * maxOfMaxJointHierarchySize);
     rules.jointAssociationSourceToTarget = (unsigned int*)  malloc(sizeof(unsigned int) * maxOfMaxJointHierarchySize);
     
     if (
          (rules.jointAssociationSourceToTargetExists) &&
          (rules.jointAssociationTargetToSourceExists) &&
          (rules.jointAssociationTargetToSource)       &&
          (rules.jointAssociationSourceToTarget)
        )
     {
      memset(rules.jointAssociationSourceToTargetExists,0,sizeof(char) * maxOfMaxJointHierarchySize);
      memset(rules.jointAssociationTargetToSourceExists,0,sizeof(char) * maxOfMaxJointHierarchySize);
      memset(rules.jointAssociationTargetToSource,0,sizeof(unsigned int) * maxOfMaxJointHierarchySize);
      memset(rules.jointAssociationSourceToTarget,0,sizeof(unsigned int) * maxOfMaxJointHierarchySize);
          
     if ( (sourceMC->jointHierarchySize<sourceMC->MAX_jointHierarchySize) && (targetMC->jointHierarchySize<targetMC->MAX_jointHierarchySize) )
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
       fprintf(stderr,"Source Joint Hierarchy has a size of %u ( max is %d )\n",sourceMC->jointHierarchySize,sourceMC->MAX_jointHierarchySize);
       fprintf(stderr,"Target Joint Hierarchy has a size of %u ( max is %d )\n",targetMC->jointHierarchySize,targetMC->MAX_jointHierarchySize);
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

   }
     if (rules.jointAssociationSourceToTargetExists) { free(rules.jointAssociationSourceToTargetExists); }
     if (rules.jointAssociationTargetToSourceExists) { free(rules.jointAssociationTargetToSourceExists); }
     if (rules.jointAssociationTargetToSource)       { free(rules.jointAssociationTargetToSource); }
     if (rules.jointAssociationSourceToTarget)       { free(rules.jointAssociationSourceToTarget); } 
     
 
     return 1;
   } else
   { fprintf(stderr,RED "Could not allocate enough space to hold the merged motion buffers..\n" NORMAL); }
 } else
 { fprintf(stderr,RED "Could not parse rule file (%s) that contains the merge mapping\n" NORMAL,pathToMergeRules); }

 return 0;
}



// -----------------------------------------

// -----------------------------------------

// -----------------------------------------

// -----------------------------------------



int bvh_mergeOffsetsInMotions(
                               struct BVH_MotionCapture * targetMC
                             )
{
    //We need to update 3 things..!
    
    //1) First allocate a brand new motion buffer that will hold the new offsets + the old rotations+offsets, and swap it for the old one..
    
    //2) We need to update numberOfValuesPerFrame motionValuesSize

    //3)We then need to update the  jointToMotionLookup/motionToJointLookup maps to their new values
       
   if (targetMC->motionValues!=0)
   {
       
       
       
       return 1; //Goal
   }
  
   return 0;
}