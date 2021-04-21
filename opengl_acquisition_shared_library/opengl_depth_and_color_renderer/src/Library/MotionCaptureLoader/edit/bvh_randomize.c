#include "bvh_randomize.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ik/hardcodedProblems_inverseKinematics.h"
#include "../mathLibrary.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */


float randomFloatA( float minVal, float maxVal )
{
  if (minVal!=maxVal) 
   { 
    if (maxVal<minVal)
    {
      float buf = minVal;
      minVal = maxVal;
      maxVal = buf;
    }

    float magnitude=maxVal-minVal;
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    float absoluteRandom = scale * magnitude;      /* [min, max] */
    float value = maxVal-absoluteRandom;
    //float value = minVal+absoluteRandom; <- same thing as above

    if (value<minVal) { fprintf(stderr,"randomFloat(%0.2f,%0.2f)=>%0.2f TOO SMALL\n",minVal,maxVal,value); } else
    if (value>maxVal) { fprintf(stderr,"randomFloat(%0.2f,%0.2f)=>%0.2f TOO BIG\n",minVal,maxVal,value); }

    return value;
  } 
  else
  {
   return minVal;
  }
}

//./BVHGUI2 --from dataset/lhand.qbvh --set 3 0.5 --set 4 -0.5 --set 5 -0.5 --set 6 0.5 --set 64 -85 --set 65 48 --set 68 25 --set 67 -43

int bvh_RandomizeBasedOnIKProblem(
                                  struct BVH_MotionCapture * mc,
                                  const char * ikProblemName
                                 )
{
   fprintf(stderr,"bvh_RandomizeBasedOnIKProblem: starting..\n");
   if (mc==0) 
       { 
          fprintf(stderr,"bvh_RandomizeBasedOnIKProblem: cannot work without a motion capture file\n");
          return 0; 
       }
   if (ikProblemName==0) 
       { 
          fprintf(stderr,"bvh_RandomizeBasedOnIKProblem: cannot work without the name of the IK problem\n");
          return 0; 
       }
       
   int success=0;
   
   //struct ikProblem tP={0};
   struct ikProblem * tP = allocateEmptyIKProblem();
   if (tP==0)
   {
       fprintf(stderr,"bvh_RandomizeBasedOnIKProblem: cannot allocateEmptyIKProblem\n");
       return 0;
   }
   
   
   fprintf(stderr,"bvh_RandomizeBasedOnIKProblem(%s)\n",ikProblemName);
   if (strcmp(ikProblemName,"lhand")==0)
   {
          fprintf(stderr,"Trying to initalize LHand.. ");
          success=prepareDefaultLeftHandProblem(
                                         tP,
                                         mc,
                                         0,//struct simpleRenderer *renderer,
                                         0,//struct MotionBuffer * previousSolution,
                                         0,//struct MotionBuffer * solution,
                                         0,//struct BVH_Transform * bvhTargetTransform,
                                         1//standalone
                                        );
          fprintf(stderr," result = %d !\n",success);
   }  else
   if (strcmp(ikProblemName,"rhand")==0)
   {
          fprintf(stderr,"Trying to initalize RHand.. ");
          success=prepareDefaultRightHandProblem(
                                         tP,
                                         mc,
                                         0,//struct simpleRenderer *renderer,
                                         0,//struct MotionBuffer * previousSolution,
                                         0,//struct MotionBuffer * solution,
                                         0,//struct BVH_Transform * bvhTargetTransform,
                                         1//standalone
                                        );
          fprintf(stderr," result = %d !\n",success); 
   } else
   {
       fprintf(stderr,"bvh_RandomizeBasedOnIKProblem: Could not identify %s ik problem!\n",ikProblemName);
       
       free(tP);
       return 0;
   }
 
  
  //Now to actually do randomizations..! 
  //-----------------------------------------------------------------------------------------------
  float * minimumRandomizationLimit = (float *) malloc(sizeof(float) * mc->numberOfValuesPerFrame);
  float * maximumRandomizationLimit = (float *) malloc(sizeof(float) * mc->numberOfValuesPerFrame);
  char * hasRandomization           = (char *)  malloc(sizeof(char)  * mc->numberOfValuesPerFrame);
  
  if ( (minimumRandomizationLimit!=0) && (maximumRandomizationLimit!=0) && (hasRandomization!=0) )
  {
   memset(minimumRandomizationLimit,0,sizeof(float) * mc->numberOfValuesPerFrame);
   memset(maximumRandomizationLimit,0,sizeof(float) * mc->numberOfValuesPerFrame);
   memset(hasRandomization,         0,sizeof(char)  * mc->numberOfValuesPerFrame);
   
   
   //unsigned int mIDOffset;
   float minimumLimit;
   float maximumLimit;
   unsigned int channelID;
   
   //First retrieve the minimum maximum limits from problem chains..!
   for (unsigned int chainID=0; chainID<tP->numberOfChains; chainID++)
    {
      for (unsigned int partID=0; partID<tP->chain[chainID].numberOfParts; partID++)
       { 
           if (tP->chain[chainID].part[partID].limits)
           {
             unsigned int jID = tP->chain[chainID].part[partID].jID; 
             fprintf(stderr,"Limits declared for => ChainID(%u) / PartID(%u) [jID=%u|%s]\n",chainID,partID,jID,mc->jointHierarchy[jID].jointName);
             
             const char * jName = mc->jointHierarchy[jID].jointName;
             
             unsigned int mIDOffset = tP->chain[chainID].part[partID].mIDStart;
             if (jID==mc->motionToJointLookup[mIDOffset].jointID)
             {
              minimumLimit = tP->chain[chainID].part[partID].minimumLimitMID[0];
              maximumLimit = tP->chain[chainID].part[partID].maximumLimitMID[0];
              if (minimumLimit>maximumLimit)
               {
                 maximumLimit = tP->chain[chainID].part[partID].minimumLimitMID[0];
                 minimumLimit = tP->chain[chainID].part[partID].maximumLimitMID[0];
               }
              channelID = mc->motionToJointLookup[mIDOffset].channelID;
              minimumRandomizationLimit[mIDOffset]=minimumLimit;
              maximumRandomizationLimit[mIDOffset]=maximumLimit;
              hasRandomization[mIDOffset]=( (maximumLimit-minimumLimit) > 0.0001);
              fprintf(stderr,"Channel #0(%s/%s)  => [%0.2f,%0.2f]\n",jName,channelNames[channelID],minimumLimit,maximumLimit);
             }
             
             mIDOffset = tP->chain[chainID].part[partID].mIDStart+1;
             if (jID==mc->motionToJointLookup[mIDOffset].jointID)
             {
              minimumLimit = tP->chain[chainID].part[partID].minimumLimitMID[1];
              maximumLimit = tP->chain[chainID].part[partID].maximumLimitMID[1];
              if (minimumLimit>maximumLimit)
               {
                 maximumLimit = tP->chain[chainID].part[partID].minimumLimitMID[1];
                 minimumLimit = tP->chain[chainID].part[partID].maximumLimitMID[1];
               }
              channelID = mc->motionToJointLookup[mIDOffset].channelID;
              minimumRandomizationLimit[mIDOffset]=minimumLimit;
              maximumRandomizationLimit[mIDOffset]=maximumLimit;
              hasRandomization[mIDOffset]=( (maximumLimit-minimumLimit) > 0.0001);
              fprintf(stderr,"Channel #1(%s/%s)  => [%0.2f,%0.2f]\n",jName,channelNames[channelID],minimumLimit,maximumLimit);
             }
             
             mIDOffset = tP->chain[chainID].part[partID].mIDEnd;
             if (jID==mc->motionToJointLookup[mIDOffset].jointID)
             {
               minimumLimit = tP->chain[chainID].part[partID].minimumLimitMID[2];
               maximumLimit = tP->chain[chainID].part[partID].maximumLimitMID[2];
               if (minimumLimit>maximumLimit)
               {
                 maximumLimit = tP->chain[chainID].part[partID].minimumLimitMID[2];
                 minimumLimit = tP->chain[chainID].part[partID].maximumLimitMID[2];
               }
               channelID = mc->motionToJointLookup[mIDOffset].channelID;
               minimumRandomizationLimit[mIDOffset]=minimumLimit;
               maximumRandomizationLimit[mIDOffset]=maximumLimit;
               hasRandomization[mIDOffset]=( (maximumLimit-minimumLimit) > 0.0001);
               fprintf(stderr,"Channel #2(%s/%s)  => [%0.2f,%0.2f]\n",jName,channelNames[channelID],minimumLimit,maximumLimit);
             }
           }
        }
    }
       
       
    //minimumRandomizationLimit and maximumRandomizationLimit 
     unsigned int fID=0;
     for (fID=0; fID<mc->numberOfFrames; fID++)
      {
       unsigned int mIDStart=fID*mc->numberOfValuesPerFrame;
       unsigned int mIDEnd=mIDStart+mc->numberOfValuesPerFrame;

       for (unsigned int mID=mIDStart; mID<mIDEnd; mID++)
         {
             unsigned int localMID = mID - mIDStart;
             
             if (hasRandomization[localMID])
                {
                  mc->motionValues[mID] = randomFloatA(minimumRandomizationLimit[localMID],maximumRandomizationLimit[localMID]); 
                  //fprintf(stderr,"mID(%u)=%f[%0.2f/%0.2f] ",mID,mc->motionValues[mID],minimumRandomizationLimit[localMID],maximumRandomizationLimit[localMID]);
                }
         }
      }
  //----------------------------------------------------------------------------------------------- 
  }
   
  if (minimumRandomizationLimit!=0) { free(minimumRandomizationLimit); }
  if (maximumRandomizationLimit!=0) { free(maximumRandomizationLimit); }
  if (hasRandomization!=0)          { free(hasRandomization);          }
  
  free(tP);
  return success;
}


int bvh_PerturbJointAnglesRange(
                                 struct BVH_MotionCapture * mc,
                                 unsigned int numberOfValues,
                                 float start,
                                 float end,
                                 unsigned int specificChannel,
                                 const char **argv,
                                 unsigned int iplus2
                                )
{

  fprintf(stderr,"\nRandomizing %u Joint Angles @ channel %u in the range [%0.2f,%0.2f] deviation\n",numberOfValues,specificChannel,start,end);
  unsigned int * selectedJoints = (unsigned int *) malloc(sizeof(unsigned int) * mc->numberOfValuesPerFrame);
  if (selectedJoints!=0)
  {
    int success=1;

    memset(selectedJoints,0,sizeof(unsigned int)* mc->numberOfValuesPerFrame);
    BVHJointID jID=0;
    fprintf(stderr,"Randomizing : ");
    unsigned int i=0;
    for (i=iplus2+1; i<=iplus2+numberOfValues; i++)
     {
      fprintf(stderr,GREEN "%s " NORMAL,argv[i]);
      if (
           //bvh_getJointIDFromJointName(
           bvh_getJointIDFromJointNameNocase(
                                       mc,
                                       argv[i],
                                       &jID
                                      )
         )
         {
           unsigned int channelsEncountered=0;
           for (unsigned int mID=0; mID<mc->numberOfValuesPerFrame; mID++)
           {
               if ( mc->motionToJointLookup[mID].jointID == jID )
               {
                 ++channelsEncountered;
                 if (specificChannel)
                 {
                   if (specificChannel==channelsEncountered)
                   {
                    selectedJoints[mID]=1;
                    fprintf(stderr,"Specific(%u) ",mID);
                   }
                 }  else
                 {
                   selectedJoints[mID]=1;
                   fprintf(stderr,"%u ",mID);
                 }
               }
           }
         } else
         {
           fprintf(stderr,RED "%s(not found) " NORMAL,argv[i]);
           success=0;
         }
     }
    fprintf(stderr,"\n");

     unsigned int fID=0;
     for (fID=0; fID<mc->numberOfFrames; fID++)
      {
       unsigned int mIDStart=fID*mc->numberOfValuesPerFrame;
       unsigned int mIDEnd=mIDStart+mc->numberOfValuesPerFrame;

       for (unsigned int mID=mIDStart; mID<mIDEnd; mID++)
         {
           if (selectedJoints[mID-mIDStart])
           {
             //fprintf(stderr,"Was %0.2f ",mc->motionValues[mID+jID]);
             mc->motionValues[mID]+=randomFloatA(start,end);
             //fprintf(stderr,"Is %0.2f ",mc->motionValues[mID+jID]);
           }
         }
      }


    free(selectedJoints);
    return success;
  }

  return 0;
}





int bvh_PerturbJointAngles(
                           struct BVH_MotionCapture * mc,
                           unsigned int numberOfValues,
                           float  deviation,
                           const char **argv,
                           unsigned int iplus2
                          )
{
  fprintf(stderr,"Asked to randomize %u Joint Angles using a %0.2f (+- %0.2f) deviation\n",numberOfValues,deviation,(float) deviation/2);
  return bvh_PerturbJointAnglesRange(
                                     mc,
                                     numberOfValues,
                                     (float) -1*deviation/2,
                                     (float) deviation/2,
                                     0,
                                     argv,
                                     iplus2
                                    );
}






int bvh_RandomizeSingleMIDInRange(
                                   struct BVH_MotionCapture * mc,
                                   BVHMotionChannelID mID,
                                   float start,
                                   float end
                                )
{
  if ( (mc!=0) && (mc->motionToJointLookup!=0) )
    {
     BVHJointID jID = mc->motionToJointLookup[mID].jointID;
     fprintf(stderr,GREEN "Asked to randomize Joint %s (jID=%u) using range %0.2f - %0.2f across %u frames\n" NORMAL,mc->jointHierarchy[jID].jointName,jID,start,end,mc->numberOfFrames);
    
     unsigned int fID=0;
     for (fID=0; fID<mc->numberOfFrames; fID++)
      {
       unsigned int mIDOfSpecificFrame=mID + fID*mc->numberOfValuesPerFrame;
 
       mc->motionValues[mIDOfSpecificFrame]=randomFloatA(start,end);
      }
      
     return 1; 
    }
   return 0;
}






int bvh_eraseJoints(
                    struct BVH_MotionCapture * mc,
                    unsigned int numberOfValues,
                    unsigned int includeEndSites,
                    const char **argv,
                    unsigned int iplus1
                   )
{
  //---------------------
  fprintf(stderr,"Asked to erase %u Joint Angles\n",numberOfValues);
  unsigned int * selectedJoints = (unsigned int *) malloc(sizeof(unsigned int) * mc->numberOfValuesPerFrame);
  if (selectedJoints!=0)
  {
    unsigned int success=1;
    memset(selectedJoints,0,sizeof(unsigned int)* mc->numberOfValuesPerFrame);
    BVHJointID jID=0;
    fprintf(stderr,"Erasing : ");
    unsigned int i=0;
    for (i=iplus1+1; i<=iplus1+numberOfValues; i++)
     {
      if (
           //bvh_getJointIDFromJointName(
           bvh_getJointIDFromJointNameNocase(
                                       mc,
                                       argv[i],
                                       &jID
                                      )
         )
         {
           fprintf(stderr,GREEN "%s " NORMAL,argv[i]);
           mc->jointHierarchy[jID].erase2DCoordinates=1;

           for (unsigned int mID=0; mID<mc->numberOfValuesPerFrame; mID++)
           {
               if (mc->motionToJointLookup[mID].jointID == jID)
               {
                selectedJoints[mID]=1;
                fprintf(stderr,"%u ",mID);
               }
           }
           //-------------------------------------------------

           if(includeEndSites)
           {
             BVHJointID jIDES=jID;
             if (bhv_jointGetEndSiteChild(mc,jID,&jIDES))
               {
                 mc->jointHierarchy[jIDES].erase2DCoordinates=1;
                 fprintf(stderr,GREEN "EndSite_%s " NORMAL,argv[i]);

                 for (unsigned int mID=0; mID<mc->numberOfValuesPerFrame; mID++)
                   {
                      if (mc->motionToJointLookup[mID].jointID == jIDES)
                         {
                           selectedJoints[mID]=1;
                           fprintf(stderr,"%u ",mID);
                         }
                  }
               }
           }

         } else
         {
           fprintf(stderr,RED "%s(not found) " NORMAL,argv[i]);
           success=0;
         }
     }
    fprintf(stderr,"\n");

     for (unsigned int fID=0; fID<mc->numberOfFrames; fID++)
      {
       unsigned int mIDStart=fID*mc->numberOfValuesPerFrame;
       unsigned int mIDEnd=mIDStart+mc->numberOfValuesPerFrame;
       for (unsigned int mID=mIDStart; mID<mIDEnd; mID++)
         {
           if (selectedJoints[mID-mIDStart])
           {
             //fprintf(stderr,"Was %0.2f ",mc->motionValues[mID+jID]);
             mc->motionValues[mID]=0.0;
             //fprintf(stderr,"Is %0.2f ",mc->motionValues[mID+jID]);
           }
         }
      }


    free(selectedJoints);
    return success;
  }

  return 0;
}




int bvh_RandomizePositionsOfFrameBasedOn3D(
                                              struct BVH_MotionCapture * mc,
                                              BVHJointID jID,
                                              BVHFrameID fID,
                                              float * minimumPosition,
                                              float * maximumPosition
                                             )
{
 if (mc!=0)
 {
  bvh_setJointPositionXAtFrame(mc,jID,fID,randomFloatA(minimumPosition[0],maximumPosition[0]));
  bvh_setJointPositionYAtFrame(mc,jID,fID,randomFloatA(minimumPosition[1],maximumPosition[1]));
  bvh_setJointPositionZAtFrame(mc,jID,fID,randomFloatA(minimumPosition[2],maximumPosition[2]));
     
  //Old code, no one guarantees the 0,1,2 offsets are correct
  //unsigned int mID=fID*mc->numberOfValuesPerFrame;
  //mc->motionValues[mID+0]=randomFloatA(minimumPosition[0],maximumPosition[0]);
  //mc->motionValues[mID+1]=randomFloatA(minimumPosition[1],maximumPosition[1]);
  //mc->motionValues[mID+2]=randomFloatA(minimumPosition[2],maximumPosition[2]); 
  return 1;     
 }

  fprintf(stderr,"bvh_RandomizePositionsBasedOn3D: Cannot be done without positional channels on root joint\n");
  return 0;
}



int bvh_RandomizePositionsBasedOn3D(
                                     struct BVH_MotionCapture * mc,
                                     float * minimumPosition,
                                     float * maximumPosition
                                    )
{
 if (mc!=0)
 {
  fprintf(stderr,"Randomizing Positions of %u frames based on 3D coordinates\n",mc->numberOfFrames);
  fprintf(stderr,"min(%0.2f,%0.2f,%0.2f) ",minimumPosition[0],minimumPosition[1],minimumPosition[2]);
  fprintf(stderr,"max(%0.2f,%0.2f,%0.2f)\n",maximumPosition[0],maximumPosition[1],maximumPosition[2]);
  
  if (mc->jointHierarchy[mc->rootJointID].hasPositionalChannels)
  {
   BVHJointID rootJID = mc->rootJointID; 
   BVHFrameID fID=0;
   
   for (fID=0; fID<mc->numberOfFrames; fID++)
    { 
     bvh_RandomizePositionsOfFrameBasedOn3D(mc,rootJID,fID,minimumPosition,maximumPosition);
    }
   return 1;    
  }
 }

  fprintf(stderr,"bvh_RandomizePositionsBasedOn3D: Cannot be done without positional channels on root joint\n");
  return 0;
}






int bvh_RandomizeRotationsOfFrameBasedOn3D(
                                               struct BVH_MotionCapture * mc,
                                               BVHJointID jID,
                                               BVHFrameID fID,
                                               float * minimumRotation,
                                               float * maximumRotation
                                          )
{

 if (mc!=0)
 { 
     //There is a dilemma here..!
     //The DAZ-friendly bvh files I use as source use a Zrotation Yrotation Xrotation channel rotation order for root joint and
     //Zrotation Xrotation Yrotation for the rest of the joints, supposedly the minimumPosition[0-3] and maximumPosition[0-3] 
     //should give information for the X rotation axis, then the Y rotation axis and then the Z rotation axis that would work 
     //regardless of rotation order, however to keep things compatible with the initial implementation I have I will just assume
     //the initial convention and will have to change this at some point in the future..  
     
     if (mc->jointHierarchy[mc->rootJointID].hasRotationalChannels)
     {
       float randomRotations[3] = {
                                    randomFloatA(minimumRotation[0],maximumRotation[0]), 
                                    randomFloatA(minimumRotation[1],maximumRotation[1]), 
                                    randomFloatA(minimumRotation[2],maximumRotation[2]) 
                                  }; 
       if (mc->jointHierarchy[mc->rootJointID].hasQuaternionRotation)
       {
         //This is a very important part of the code..
         //We assume a ZYX Rotation order we have inherited from https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/daz-friendly-release
         float randomQuaternion[4]; 
        
        //TODO: this or not todo ?
        //fprintf(stderr,YELLOW "Maybe the random rotations that are converted to a quaternion don't need to be swapped?\n" NORMAL ); 
        float buffer = randomRotations[0];
        randomRotations[0]=randomRotations[2];
        randomRotations[2]=buffer;
        
        //BVH Quaternion
        euler2Quaternions(randomQuaternion,randomRotations,qWqXqYqZ); 
        bvh_setJointRotationWAtFrame(mc,jID,fID,randomQuaternion[0]);
        bvh_setJointRotationXAtFrame(mc,jID,fID,randomQuaternion[1]);
        bvh_setJointRotationYAtFrame(mc,jID,fID,randomQuaternion[2]);
        bvh_setJointRotationZAtFrame(mc,jID,fID,randomQuaternion[3]); 
       } else
       {
        bvh_setJointRotationZAtFrame(mc,jID,fID,randomRotations[0]);
        bvh_setJointRotationYAtFrame(mc,jID,fID,randomRotations[1]);
        bvh_setJointRotationXAtFrame(mc,jID,fID,randomRotations[2]);
       }
     }
     
   return 1;  
  }
  
 fprintf(stderr,"bvh_RandomizeRotationsOfFrameBasedOn3D: Cannot be done without positional channels on root joint\n");
 return 0;
}







int bvh_RandomizeRotationsBasedOn3D(
                                     struct BVH_MotionCapture * mc,
                                     float * minimumRotation,
                                     float * maximumRotation
                                    )
{
 if (mc!=0)
 {
  fprintf(stderr,"Randomizing Rotations of %u frames based on 3D coordinates\n",mc->numberOfFrames);
  fprintf(stderr,"min(%0.2f,%0.2f,%0.2f)",minimumRotation[0],minimumRotation[1],minimumRotation[2]);
  fprintf(stderr,"max(%0.2f,%0.2f,%0.2f)",maximumRotation[0],maximumRotation[1],maximumRotation[2]);
  
  if (mc->jointHierarchy[mc->rootJointID].hasRotationalChannels)
  {
   BVHJointID rootJID = mc->rootJointID; 
   BVHFrameID fID=0;
   
   for (fID=0; fID<mc->numberOfFrames; fID++)
    {  
      bvh_RandomizeRotationsOfFrameBasedOn3D(mc,rootJID,fID,minimumRotation,maximumRotation);
    }
   return 1;
  } 
 }
 
 fprintf(stderr,"bvh_RandomizeRotationsBasedOn3D: Cannot be done without positional channels on root joint\n");
 return 0;
}




int bvh_RandomizePositionRotation(
                                  struct BVH_MotionCapture * mc,
                                  float * minimumPosition,
                                  float * minimumRotation,
                                  float * maximumPosition,
                                  float * maximumRotation
                                 )
{
  return ( 
            (bvh_RandomizePositionsBasedOn3D(mc,minimumPosition,maximumPosition)) &&
            (bvh_RandomizeRotationsBasedOn3D(mc,minimumPosition,maximumPosition)) 
         );
}



int bvh_RandomizePositionRotation2Ranges(
                                         struct BVH_MotionCapture * mc,
                                         float * minimumPositionRangeA,
                                         float * minimumRotationRangeA,
                                         float * maximumPositionRangeA,
                                         float * maximumRotationRangeA,
                                         float * minimumPositionRangeB,
                                         float * minimumRotationRangeB,
                                         float * maximumPositionRangeB,
                                         float * maximumRotationRangeB
                                        )
{
  fprintf(stderr,"Randomizing %u frames at two ranges\n",mc->numberOfFrames);
  fprintf(stderr,"Range A\n");
  fprintf(stderr,"min(Pos[%0.2f,%0.2f,%0.2f],",minimumPositionRangeA[0],minimumPositionRangeA[1],minimumPositionRangeA[2]);
  fprintf(stderr,"Rot[%0.2f,%0.2f,%0.2f])\n",minimumRotationRangeA[0],minimumRotationRangeA[1],minimumRotationRangeA[2]);
  fprintf(stderr,"max(Pos[%0.2f,%0.2f,%0.2f],",maximumPositionRangeA[0],maximumPositionRangeA[1],maximumPositionRangeA[2]);
  fprintf(stderr,"Rot[%0.2f,%0.2f,%0.2f])\n",maximumRotationRangeA[0],maximumRotationRangeA[1],maximumRotationRangeA[2]);
  fprintf(stderr,"Range B\n");
  fprintf(stderr,"min(Pos[%0.2f,%0.2f,%0.2f],",minimumPositionRangeB[0],minimumPositionRangeB[1],minimumPositionRangeB[2]);
  fprintf(stderr,"Rot[%0.2f,%0.2f,%0.2f])\n",minimumRotationRangeB[0],minimumRotationRangeB[1],minimumRotationRangeB[2]);
  fprintf(stderr,"max(Pos[%0.2f,%0.2f,%0.2f],",maximumPositionRangeB[0],maximumPositionRangeB[1],maximumPositionRangeB[2]);
  fprintf(stderr,"Rot[%0.2f,%0.2f,%0.2f])\n",maximumRotationRangeB[0],maximumRotationRangeB[1],maximumRotationRangeB[2]);

  //fprintf(stderr,"Exiting\n");
  //exit(0);
  BVHJointID rootJID = mc->rootJointID; 
  BVHFrameID fID=0;
   
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;
   float whichHalf = rand() / (float) RAND_MAX; /* [0, 1.0] */

   if (whichHalf<0.5)
           { 
             bvh_RandomizeRotationsOfFrameBasedOn3D(mc,rootJID,fID,minimumRotationRangeA,maximumRotationRangeA); 
             bvh_RandomizePositionsOfFrameBasedOn3D(mc,rootJID,fID,minimumPositionRangeA,maximumPositionRangeA); 
           } else
           {
             bvh_RandomizeRotationsOfFrameBasedOn3D(mc,rootJID,fID,minimumRotationRangeB,maximumRotationRangeB); 
             bvh_RandomizePositionsOfFrameBasedOn3D(mc,rootJID,fID,minimumPositionRangeB,maximumPositionRangeB); 
           }
  }
 return 1;
}



void transform2DFProjectedPointTo3DPoint(float fX,float fY,float cX,float cY,unsigned int width,unsigned int height,
                                        float x2D , float y2D  , float depthValue ,
                                        float * x3D , float * y3D)
{
 *x3D = (float) (x2D - cX) * (depthValue / fX);
 *y3D = (float) (y2D - cY) * (depthValue / fY);
}






int bvh_RandomizePositionFrom2D(
                                 struct BVH_MotionCapture * mc,
                                 float * minimumRotation,
                                 float * maximumRotation,
                                 float minimumDepth,float maximumDepth,
                                 float fX,float fY,float cX,float cY,unsigned int width,unsigned int height
                                )
{
  fprintf(stderr,"Randomizing %u frames  using 2D randomizations \n",mc->numberOfFrames);
  
  if (mc->jointHierarchy[mc->rootJointID].hasQuaternionRotation)
          {
              fprintf(stderr,"TODO: Handle quaternion rotation here..!\n");
          }


  unsigned int borderX=width/7; //8
  unsigned int borderY=height/4;//5

  float positionX,positionY,positionZ;


  BVHJointID rootJID = mc->rootJointID; 
  BVHFrameID fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;

   
   positionZ = randomFloatA(minimumDepth,maximumDepth);
   //mc->motionValues[mID+2]=randomFloatA(minimumDepth,maximumDepth);
   unsigned int x2D = borderX+ rand()%(width-borderX*2);
   unsigned int y2D = borderY+ rand()%(height-borderY*2);
   //void transform2DFProjectedPointTo3DPoint(float fX,float fY,float cX,float cY,unsigned int width,unsigned int height,float x2D,float y2D,float depthValue,float * x3D,float * y3D)
   //transform2DFProjectedPointTo3DPoint(fX,fY,cX,cY,width,height,(float) x2D,(float) y2D,mc->motionValues[mID+2],&mc->motionValues[mID+0],&mc->motionValues[mID+1]);
   transform2DFProjectedPointTo3DPoint(fX,fY,cX,cY,width,height,(float) x2D,(float) y2D,positionZ,&positionX,&positionY);

   //Set random points..
   bvh_setJointPositionXAtFrame(mc,rootJID,fID,positionX);
   bvh_setJointPositionYAtFrame(mc,rootJID,fID,positionY);
   bvh_setJointPositionZAtFrame(mc,rootJID,fID,positionZ);

   //mc->motionValues[mID+3]=randomFloatA(minimumRotation[0],maximumRotation[0]);
   //mc->motionValues[mID+4]=randomFloatA(minimumRotation[1],maximumRotation[1]);
   //mc->motionValues[mID+5]=randomFloatA(minimumRotation[2],maximumRotation[2]);
  }
 
 bvh_RandomizeRotationsBasedOn3D(mc,minimumRotation,maximumRotation);
 return 1;
}


int bvh_RandomizePositionFrom2DRotation2Ranges(
                                               struct BVH_MotionCapture * mc,
                                               float * minimumRotationRangeA,
                                               float * maximumRotationRangeA,
                                               float * minimumRotationRangeB,
                                               float * maximumRotationRangeB,
                                               float minimumDepth,float maximumDepth,
                                               float fX,float fY,float cX,float cY,unsigned int width,unsigned int height
                                              )
{
  fprintf(stderr,"Randomizing %u frames at two ranges\n",mc->numberOfFrames);

  //Randomize Positions using same codepath as bvh_RandomizePositionFrom2D (we also overwrite rotations but will get randomized again anyway)
  bvh_RandomizePositionFrom2D(
                               mc,
                               minimumRotationRangeA,
                               maximumRotationRangeA,
                               minimumDepth,maximumDepth,
                               fX,fY,cX,cY,width,height
                              );

  //Just Randomize Rotations like bvh_RandomizePositionRotation2Ranges
  BVHJointID rootJID = mc->rootJointID; 
  BVHFrameID fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  { 
   float whichHalf = rand() / (float) RAND_MAX; /* [0, 1.0] */

   if (whichHalf<0.5)
           {
              bvh_RandomizeRotationsOfFrameBasedOn3D(mc,rootJID,fID,minimumRotationRangeA,maximumRotationRangeA);
           } else
           {
              bvh_RandomizeRotationsOfFrameBasedOn3D(mc,rootJID,fID,minimumRotationRangeB,maximumRotationRangeB);
           }
  }
 return 1;
}




/*
int bvh_TestRandomizationLimitsXYZ(
                                   struct BVH_MotionCapture * mc,
                                   float * minimumPosition,
                                   float * maximumPosition
                                  )
{
  if (mc->numberOfFrames<8)
  {
    return 0;
  }
  mc->numberOfFrames=8;

  unsigned int fID,mID;
  //----------------------------------------
  fID=0; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=minimumPosition[0];  mc->motionValues[mID+1]=minimumPosition[1]; mc->motionValues[mID+2]=minimumPosition[2]; //Minimum X , Minimum Y , Minimum Z
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=minimumPosition[0]; mc->motionValues[mID+1]=minimumPosition[1];  mc->motionValues[mID+2]=maximumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=minimumPosition[0]; mc->motionValues[mID+1]=maximumPosition[1];  mc->motionValues[mID+2]=minimumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=minimumPosition[0]; mc->motionValues[mID+1]=maximumPosition[1];  mc->motionValues[mID+2]=maximumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=maximumPosition[0]; mc->motionValues[mID+1]=minimumPosition[1];  mc->motionValues[mID+2]=minimumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=maximumPosition[0]; mc->motionValues[mID+1]=minimumPosition[1];  mc->motionValues[mID+2]=maximumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=maximumPosition[0]; mc->motionValues[mID+1]=maximumPosition[1];  mc->motionValues[mID+2]=minimumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=maximumPosition[0]; mc->motionValues[mID+1]=maximumPosition[1];  mc->motionValues[mID+2]=maximumPosition[2]; //Maximum X , Maximum Y, Maximum Z


  return 1;
}
*/