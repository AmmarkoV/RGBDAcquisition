#include "bvh_merge.h"
#include "../bvh_to_tri_pose.h"
#include "bvh_cut_paste.h"


#include <stdio.h>
#include <stdlib.h>
// ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --merge Motions/49_04.bvh Motions/mergeDazFriendlyMoreThan19.profile --bvh testMerge.bvh


//  ./BVHTester --from Motions/DAZFriendlyCMUPlusHead.bvh --print --merge Motions/02_03.bvh Motions/mergeDazFriendlyAndAddHead.profile --bvh testMerge.bvh
int bvh_mergeWith(
                   struct BVH_MotionCapture * mc,
                   struct BVH_MotionCapture * mcToMerge,
                   const char * pathToMergeRules
                 )
{
 fprintf(stderr,"bvh_mergeWith will try to merge the skeleton and motion frames of file %s in original file (%s) \n",mcToMerge->fileName,mc->fileName);
 struct bvhToTRI bvhtri={0};
 if (bvh_loadBVHToTRIAssociationFile(pathToMergeRules,&bvhtri,mc))
 {
   if (
        bvh_EraseAndAllocateSpaceForNumberOfFrames(
                                                   mc,
                                                   mcToMerge->numberOfFramesEncountered
                                                  )
      )
   {
     //Our target file has enough space to accomodate the incoming motion buffers..!
     if (mcToMerge->numberOfFrames!=mc->numberOfFrames)
     {
      fprintf(stderr,"CodeBug: we don't have enough space to write\n");
     }

     unsigned int frameID=0;
     for (frameID=0; frameID<mcToMerge->numberOfFrames; frameID++)
     {



     }

     return 0;
   } else
   { fprintf(stderr,"Could not allocate enough space to hold the merged motion buffers..\n"); }
 } else
 { fprintf(stderr,"Could not parse rule file (%s) that contains the merge mapping\n",pathToMergeRules); }

 return 0;
}
