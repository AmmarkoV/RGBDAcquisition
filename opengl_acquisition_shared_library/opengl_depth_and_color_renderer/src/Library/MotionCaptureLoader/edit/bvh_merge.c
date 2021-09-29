#include "bvh_merge.h"
#include "bvh_rename.h"
#include "../import/fromBVH.h"
#include "../export/bvh_to_bvh.h"
#include "../calculate/bvh_to_tri_pose.h"
#include "bvh_cut_paste.h"

#include "cTextFileToMemory.h"

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
       fprintf(stderr,"Source Joint Hierarchy has a size of %u ( max is %u )\n",sourceMC->jointHierarchySize,sourceMC->MAX_jointHierarchySize);
       fprintf(stderr,"Target Joint Hierarchy has a size of %u ( max is %u )\n",targetMC->jointHierarchySize,targetMC->MAX_jointHierarchySize);
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

 
int bvh_updateJointLookupMaps(struct BVH_MotionCapture * mc)
{
  // Make sure the max joint hierarchy is in sync with the current joint hierarchy
  if (mc->MAX_jointHierarchySize < mc->jointHierarchySize)
     {
      fprintf(stderr,YELLOW "increasing MAX_jointHierarchySize from %u to %u ..\n" NORMAL,mc->MAX_jointHierarchySize,mc->jointHierarchySize);
      mc->MAX_jointHierarchySize = mc->jointHierarchySize;
     }
   //-------------------------------------------------------------------------------
    
    
    
   //Recount required motion values, they might have increased.. 
   fprintf(stderr,"Total Joints %u \n",mc->jointHierarchySize);
   unsigned int totalChannelsRecounted=0;
   for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
            {
              //We can now update lookup tables..
              totalChannelsRecounted += mc->jointHierarchy[jID].loadedChannels;
              //fprintf(stderr,RED " %u " NORMAL,mc->jointHierarchy[jID].loadedChannels);
            }
   if (totalChannelsRecounted>mc->motionValuesSize)
      {
       fprintf(stderr,YELLOW "increasing Total Motion Channels from %u to %u \n" NORMAL,mc->motionValuesSize,totalChannelsRecounted); 
       mc->motionValuesSize = totalChannelsRecounted;
       mc->numberOfValuesPerFrame = totalChannelsRecounted;
      }
   //-------------------------------------------------------------------------------


  //Free previous joint lookups
  //-------------------------------------------------------------------------------
  if  (mc->jointToMotionLookup!=0)  { free(mc->jointToMotionLookup); mc->jointToMotionLookup=0;}
  mc->jointToMotionLookup  = (struct BVH_JointToMotion_LookupTable *) malloc( sizeof(struct BVH_JointToMotion_LookupTable) * (mc->MAX_jointHierarchySize+1) );
  //-------------------------------------------------------------------------------
  if  (mc->motionToJointLookup!=0)  { free(mc->motionToJointLookup); mc->motionToJointLookup=0;}
  mc->motionToJointLookup  = (struct BVH_MotionToJoint_LookupTable *) malloc( sizeof(struct BVH_MotionToJoint_LookupTable) * (mc->motionValuesSize+1) );
  //-------------------------------------------------------------------------------
  if  (mc->motionValues!=0)  { free(mc->motionValues); mc->motionValues=0;}
  mc->motionValues = (float*)  malloc(sizeof(float) * (1+mc->motionValuesSize));
  //-------------------------------------------------------------------------------
 
  if (
       (mc->jointToMotionLookup!=0) && (mc->motionToJointLookup!=0)
     )
     {
       //Clean everything..! 
       memset(mc->jointToMotionLookup,0,sizeof(struct BVH_JointToMotion_LookupTable) * mc->MAX_jointHierarchySize);
       memset(mc->motionToJointLookup,0,sizeof(struct BVH_MotionToJoint_LookupTable) * mc->motionValuesSize);

       //Repopulate everything..!
       BVHMotionChannelID mID=0;
       for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
            {
              //We can now update lookup tables..
              for (unsigned int cL=0; cL<mc->jointHierarchy[jID].loadedChannels; cL++)
                       { 
                         if (mc->motionValuesSize <= mID)
                         {
                             fprintf(stderr,RED "Not enough memory for motion to joint lookup table..\n" NORMAL);
                             return 0; 
                         }
                         
                         //For each declared channel we need to enumerate the label to a value
                         unsigned int thisChannelID = mc->jointHierarchy[jID].channelType[cL];
                         
                         //Update jointToMotion Lookup Table..
                         mc->jointToMotionLookup[jID].channelIDMotionOffset[thisChannelID] = mID; 

                         //Update motionToJoint Lookup Table..
                         mc->motionToJointLookup[mID].channelID = thisChannelID;
                         mc->motionToJointLookup[mID].jointID   = jID;
                         mc->motionToJointLookup[mID].parentID  = mc->jointHierarchy[jID].parentJoint; 

                         ++mID;
                       }
            }

       return 1;
     }

  //---------------------
  fprintf(stderr,RED "Failed allocating memory, giving up on bvh_updateJointLookupMaps\n" NORMAL);  
  return 0;
}





int bvh_expandPositionalChannelsOfSelectedJoints(struct BVH_MotionCapture * mc)
{
 if ( (mc!=0) && (mc->selectedJoints!=0) && (mc->jointHierarchy!=0) )
 {
    for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
              {
                  if (mc->selectedJoints[jID])
                  {
                     //fprintf(stderr,"JointSelected(%s,rotOrder:%u) ",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].channelRotationOrder);
                     if (!mc->jointHierarchy[jID].hasPositionalChannels)
                     {
                         for (unsigned int channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
                         { //Shift all channels 3 positions down
                           mc->jointHierarchy[jID].channelType[3+channelID] = mc->jointHierarchy[jID].channelType[channelID];
                         }
                         //mc->jointHierarchy[jID].channelType[3] = mc->jointHierarchy[jID].channelType[0];
                         //mc->jointHierarchy[jID].channelType[4] = mc->jointHierarchy[jID].channelType[1];
                         //mc->jointHierarchy[jID].channelType[5] = mc->jointHierarchy[jID].channelType[2];
                         
                         //Finally add the positional component..
                         mc->jointHierarchy[jID].channelType[0] = BVH_POSITION_X;
                         mc->jointHierarchy[jID].channelType[1] = BVH_POSITION_Y;
                         mc->jointHierarchy[jID].channelType[2] = BVH_POSITION_Z;

                         // Set Positional Channel..
                         mc->jointHierarchy[jID].hasPositionalChannels = 1;

                         //Add to loaded channels..
                         mc->jointHierarchy[jID].loadedChannels+=3;
                     }
                  }
              }
              
  //Updated joint lookup maps..
  return bvh_updateJointLookupMaps(mc); 
 }
 
 return 0;
}



// We want to use a neutral BVH file as our "master" BVH  and basically perform the following 2 operations
// First of all extend all motion vectors of "parentJoint" and all their children adding Positional X Y Z coordinates
// Second use the offsets of the neutral file as zero and resolve all offsets of BVH files in pathToListOfFiles so that they share the same
// HIERARCHY Offsets and have all their differences "explained" by the MOTION 
// We basically want the same HIERARCHY across all files and we want to encode the different offsets on the MOTION part of the BVH files..
//
// Use : 
//          ./BVHTester --headrobot dataset/faces/neutral.bvh Head dataset/faces/list.txt dataset/faces/
//
//
int bvh_mergeFacesRobot(int startAt,int argc,const char **argv)
{
  char filename[1024]={0};
  const char * pathToNeutralFile = argv[startAt];
  const char * parentJoint = argv[startAt+1];
  const char * pathToListOfFiles = argv[startAt+2];
  const char * pathToPrependToFilesOfList = argv[startAt+3];
    
  fprintf(stderr,"mergeFacesRobot, Neutral BVH file : %s ",pathToNeutralFile);
  fprintf(stderr,"mergeFacesRobot, Parent Joint : %s ",parentJoint);
  fprintf(stderr,"mergeFacesRobot, List of files : %s ",pathToListOfFiles);
  struct cTextFileToMemory bvhfiles;
  struct BVH_MotionCapture bvhNeutralFile={0};
  struct BVH_MotionCapture bvhFaceFileOriginal={0};
  struct BVH_MotionCapture bvhFaceFileToBeMerged={0};
  float scaleWorld = 1.0; 
  
   if (!bvh_loadBVH(pathToNeutralFile, &bvhNeutralFile, scaleWorld))
          {
            fprintf(stderr,"Error loading bvh file..\n");
            return 0;
          }
   bvh_renameJointsForCompatibility(&bvhNeutralFile);
   bvh_selectChildrenOfJoint(&bvhNeutralFile,parentJoint);
    
   if ( ctftm_loadTextFileToMemory(&bvhfiles,pathToListOfFiles) )
    { 
      fprintf(stderr,"Found %u records in it\n",ctftm_getNumberOfRecords(&bvhfiles));
      for (int i=0; i<ctftm_getNumberOfRecords(&bvhfiles); i++)
      {
          char * record = ctftm_getRecords(&bvhfiles,i);
          if ( (record!=0) && (strlen(record)>1) )
          {
            snprintf(filename,1024,"%s/%s",pathToPrependToFilesOfList,record);
            fprintf(stderr,"Record %u = Value `%s`\n",i,filename);
          
           
            if (
                 (!bvh_loadBVH(filename,&bvhFaceFileToBeMerged,scaleWorld)) ||
                 (!bvh_loadBVH(filename,&bvhFaceFileOriginal,scaleWorld))
               )
            {
             fprintf(stderr,"Error loading bvh file %s ..\n",filename);
             break;
            } else
            {
              //Complete loading of BVH files by doing renames and selecting the joints we are
              //interested in  ..
              bvh_renameJointsForCompatibility(&bvhFaceFileOriginal);
              bvh_selectChildrenOfJoint(&bvhFaceFileOriginal,parentJoint);
              //-----------------------------------------------------------------------
              bvh_renameJointsForCompatibility(&bvhFaceFileToBeMerged);
              bvh_selectChildrenOfJoint(&bvhFaceFileToBeMerged,parentJoint);
              //-----------------------------------------------------------------------
              //-----------------------------------------------------------------------
              
              
              if ( 
                   (bvhNeutralFile.jointHierarchySize==bvhFaceFileOriginal.jointHierarchySize) &&
                   (bvhNeutralFile.jointHierarchySize==bvhFaceFileToBeMerged.jointHierarchySize)
                 )
              {
               if ( bvh_expandPositionalChannelsOfSelectedJoints(&bvhFaceFileToBeMerged) )
               {
                //From here on bvhFaceFileToBeMerged also has the new positional channels allocated and enabled after joint "parentJoint"
                //bvhFaceFileOriginal has its original layout.. Last step is to copy the original channels to their correct targets..!
                
                BVHMotionChannelID mIDOriginal=0;
                BVHMotionChannelID mIDMerged=0;
                for (BVHJointID jID=0; jID<bvhFaceFileToBeMerged.jointHierarchySize; jID++)
                {
                  if ( (bvhFaceFileOriginal.selectedJoints[jID]) && (bvhFaceFileToBeMerged.selectedJoints[jID]) )
                  {
                     //The only way for loaded channels to differ is because of the bvh_expandPositionalChannelsOfSelectedJoints call..
                     if ( bvhFaceFileOriginal.jointHierarchy[jID].loadedChannels != bvhFaceFileToBeMerged.jointHierarchy[jID].loadedChannels )
                     {
                      //Ok this is one of the affected joints that we added positional channels to..!
                      //There where no positional channels in the bvhFaceFileOriginal so what we do is we will subtract the original XYZ offsets from the neutral XYZ offsets

                      //X Position ______________________________________________________________________________________________________________________________________________
                      if (mIDMerged<bvhFaceFileToBeMerged.numberOfValuesPerFrame)
                      {
                        bvhFaceFileToBeMerged.motionValues[mIDMerged]       = bvhFaceFileOriginal.jointHierarchy[jID].offset[0] - bvhNeutralFile.jointHierarchy[jID].offset[0];
                        bvhFaceFileToBeMerged.jointHierarchy[jID].offset[0] = bvhNeutralFile.jointHierarchy[jID].offset[0];
                        ++mIDMerged;
                      } else { fprintf(stderr,RED "Overflow at jointID %u / mID %u \n" NORMAL,jID,mIDMerged); break; }
                      
                      //Y Position ______________________________________________________________________________________________________________________________________________
                      if (mIDMerged<bvhFaceFileToBeMerged.numberOfValuesPerFrame)
                      {
                        bvhFaceFileToBeMerged.motionValues[mIDMerged]       = bvhFaceFileToBeMerged.jointHierarchy[jID].offset[1] - bvhNeutralFile.jointHierarchy[jID].offset[1]; 
                        bvhFaceFileToBeMerged.jointHierarchy[jID].offset[1] = bvhNeutralFile.jointHierarchy[jID].offset[1];
                        ++mIDMerged;
                      } else { fprintf(stderr,RED "Overflow at jointID %u / mID %u \n" NORMAL,jID,mIDMerged); break; }
                      
                      //Z Position ______________________________________________________________________________________________________________________________________________
                      if (mIDMerged<bvhFaceFileToBeMerged.numberOfValuesPerFrame)
                      {
                      bvhFaceFileToBeMerged.motionValues[mIDMerged]       = bvhFaceFileToBeMerged.jointHierarchy[jID].offset[2] - bvhNeutralFile.jointHierarchy[jID].offset[2]; 
                      bvhFaceFileToBeMerged.jointHierarchy[jID].offset[2] = bvhNeutralFile.jointHierarchy[jID].offset[2];
                      ++mIDMerged;
                      } else { fprintf(stderr,RED "Overflow at jointID %u / mID %u \n" NORMAL,jID,mIDMerged); break; }


                      //Refresh 4x4 offset matrix
                      bvh_populateStaticTransformationOfJoint(&bvhFaceFileToBeMerged,jID);
                      
                      
                      unsigned int originalLoadedChannels =  bvhFaceFileOriginal.jointHierarchy[jID].loadedChannels;
                      if (
                            (mIDOriginal + originalLoadedChannels <= bvhFaceFileOriginal.numberOfValuesPerFrame) &&
                            (mIDMerged   + originalLoadedChannels <= bvhFaceFileToBeMerged.numberOfValuesPerFrame)
                         )
                      {
                       //The final loadedChannels of the original bvh file get copied
                       for (unsigned int channelID=0; channelID<originalLoadedChannels; channelID++)
                        {
                         fprintf(stderr,"bvhFaceFileToBeMerged.motionValues[%u/%u] = bvhFaceFileOriginal.motionValues[%u/%u];\n",mIDMerged,bvhFaceFileToBeMerged.numberOfValuesPerFrame,mIDOriginal,bvhFaceFileOriginal.numberOfValuesPerFrame);
                         bvhFaceFileToBeMerged.motionValues[mIDMerged] = bvhFaceFileOriginal.motionValues[mIDOriginal];   
                         ++mIDMerged;  
                         ++mIDOriginal;
                        }
                      }  else
                     { 
                        fprintf(stderr,RED "Overflow at final copy jointID %u \n" NORMAL,jID); 
                        fprintf(stderr,RED "Original mID %u -> %u / %u \n" NORMAL,mIDOriginal,mIDOriginal+originalLoadedChannels,bvhFaceFileOriginal.numberOfValuesPerFrame); 
                        fprintf(stderr,RED "Merged mID %u -> %u / %u \n" NORMAL,mIDMerged,mIDMerged+originalLoadedChannels,bvhFaceFileToBeMerged.numberOfValuesPerFrame); 
                        break; 
                     }
                    }
                  }
                   else
                  {
                     //This is not an altered join so we just need perform 1:1 copy of the loaded channels..
                     //We will first perform an array bounds check to make sure we won't write/read incorrectly..
                     unsigned int originalLoadedChannels =  bvhFaceFileOriginal.jointHierarchy[jID].loadedChannels;
                     if (
                            (mIDOriginal + originalLoadedChannels <= bvhFaceFileOriginal.numberOfValuesPerFrame) &&
                            (mIDMerged   + originalLoadedChannels <= bvhFaceFileToBeMerged.numberOfValuesPerFrame)
                         )
                    {
                      //We are in bounds so copy the channels 1:1
                      for (unsigned int channelID=0; channelID<originalLoadedChannels; channelID++)
                      {
                         bvhFaceFileToBeMerged.motionValues[mIDMerged] = bvhFaceFileOriginal.motionValues[mIDOriginal];   
                         ++mIDMerged;  
                         ++mIDOriginal;
                      }
                    }  else 
                    { 
                        fprintf(stderr,RED "Overflow at 1:1 copy jointID %u \n" NORMAL,jID); 
                        fprintf(stderr,RED "Original mID %u -> %u / %u \n" NORMAL,mIDOriginal,mIDOriginal+originalLoadedChannels,bvhFaceFileOriginal.numberOfValuesPerFrame); 
                        fprintf(stderr,RED "Merged mID %u -> %u / %u \n" NORMAL,mIDMerged,mIDMerged+originalLoadedChannels,bvhFaceFileToBeMerged.numberOfValuesPerFrame); 
                        break; 
                     }
                  }
                }
                
                //Success adding positional channels..
                snprintf(filename,1024,"%s/merged_%s",pathToPrependToFilesOfList,record);
                dumpBVHToBVH(
                             filename,
                             &bvhFaceFileToBeMerged
                            ); 
              } else
              {
                  fprintf(stderr,RED "Failed to expand BVH joint hierarchy using bvh_expandPositionalChannelsOfSelectedJoints.. !\n" NORMAL);
              }
             } else
             {
                fprintf(stderr,RED "Mismatching BVH joint hierarchy sizes.. !\n" NORMAL);
                fprintf(stderr,RED "Neutral BVH file has %u joints (%s).. !\n" NORMAL,bvhNeutralFile.jointHierarchySize,pathToNeutralFile);
                fprintf(stderr,RED "Original BVH file has %u joints (%s).. !\n" NORMAL,bvhFaceFileOriginal.jointHierarchySize,filename);
                fprintf(stderr,RED "Merged BVH file has %u joints (%s).. !\n" NORMAL,bvhFaceFileToBeMerged.jointHierarchySize,filename); 
             }
             //-----------------------------------------------------------------------
             //-----------------------------------------------------------------------
             //-----------------------------------------------------------------------
           
             bvh_free(&bvhFaceFileOriginal);
             bvh_free(&bvhFaceFileToBeMerged);
            } 
           
           
           
           //Done with file..
          }
      }
    }

  bvh_free(&bvhNeutralFile);
  return 1;
}


// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------

int bvh_mergeOffsetsInMotions(
                               struct BVH_MotionCapture * mc
                             )
{  
    // Test : ./GroundTruthDumper --from dataset/head.bvh --addpositionalchannels --bvh test.bvh

    //We need to update 3 things..!
    
    //1) First allocate a brand new motion buffer that will hold the new offsets + the old rotations+offsets, and swap it for the old one..
    
    //2) We need to update numberOfValuesPerFrame motionValuesSize

    //3)We then need to update the  jointToMotionLookup/motionToJointLookup maps to their new values
       
   if (mc->motionValues!=0)
   {
       unsigned int newNumberOfValuesPerFrame = 0;
       unsigned int newMotionValuesSize=0;
       for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
       {
           newNumberOfValuesPerFrame+=mc->jointHierarchy[jID].loadedChannels;
       }
       fprintf(stderr,"BVH file will be resized to accomodate %u channels instead of the old %u\n",newNumberOfValuesPerFrame,mc->numberOfValuesPerFrame); 
       
       newMotionValuesSize=newNumberOfValuesPerFrame*mc->numberOfFrames;
       fprintf(stderr,"Total size will be adjusted to %u instead of the old %u\n",newMotionValuesSize,mc->motionValuesSize); 
       
       float * newMotionValues = (float*) malloc(sizeof(float) * newMotionValuesSize);
       if (newMotionValues!=0)
       {
         //
         BVHMotionChannelID oldMID=0,newMID=0;
         for (unsigned int fID=0; fID<mc->numberOfFrames; fID++)
          {
           for (BVHJointID jID=0; jID<mc->jointHierarchySize; jID++)
            {
              //All joints will gain a positional channel
              if (!mc->jointHierarchy[jID].hasPositionalChannels) 
              {
                  //If this joint did not have a positional component we will add it..! 
                  newMotionValues[newMID] = mc->jointHierarchy[jID].offset[0]; ++newMID;
                  newMotionValues[newMID] = mc->jointHierarchy[jID].offset[1]; ++newMID;
                  newMotionValues[newMID] = mc->jointHierarchy[jID].offset[2]; ++newMID;
                  mc->jointHierarchy[jID].hasPositionalChannels = 1;
              }

             //Copy existing information..
             for (unsigned int channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
               {
                newMotionValues[newMID] = mc->motionValues[oldMID];   ++oldMID;  ++newMID;
               }
            } 
          }
         
         //Deallocate old buffer..
         free(mc->motionValues);
         
         //Update everything to new data..
         mc->motionValues = newMotionValues;
         mc->motionValuesSize = newMotionValuesSize;
         mc->numberOfValuesPerFrame = newNumberOfValuesPerFrame; 
         
         if ( bvh_updateJointLookupMaps(mc) )
         {
           return 1; //Goal 
         }
         
       }
   }
  
   return 0;
}
