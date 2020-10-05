/*
    Written by Ammar Qammaz a.k.a. AmmarkoV 2018
    --
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

#include <stdio.h>
#include <math.h>
#include "bvh_loader.h"
#include "edit/bvh_rename.h"
#include "../TrajectoryParser/InputParser_C.h"

#include "edit/bvh_remapangles.h"
#include "edit/bvh_cut_paste.h"

#include "import/fromBVH.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */
#define BOLDBLACK   "\033[1m\033[30m"      /* Bold Black */
#define BOLDRED     "\033[1m\033[31m"      /* Bold Red */
#define BOLDGREEN   "\033[1m\033[32m"      /* Bold Green */
#define BOLDYELLOW  "\033[1m\033[33m"      /* Bold Yellow */
#define BOLDBLUE    "\033[1m\033[34m"      /* Bold Blue */
#define BOLDMAGENTA "\033[1m\033[35m"      /* Bold Magenta */
#define BOLDCYAN    "\033[1m\033[36m"      /* Bold Cyan */
#define BOLDWHITE   "\033[1m\033[37m"      /* Bold White */





//A very brief documentation of the BVH spec :
//http://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html?fbclid=IwAR0BopXj4Kft_RAEE41VLblkkPGHVF8-mon3xSCBMZueRtyb9LCSZDZhXPA

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        HIERARCHY PARSING
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------


int enumerateChannelOrderFromTypes(char typeA,char typeB,char typeC)
{
  switch (typeA)
  {
    case BVH_ROTATION_X :
         switch (typeB)
         {
           case BVH_ROTATION_Y :
               if (typeC == BVH_ROTATION_Z) { return BVH_ROTATION_ORDER_XYZ; }
           break;
           case BVH_ROTATION_Z :
               if (typeC == BVH_ROTATION_Y) { return BVH_ROTATION_ORDER_XZY; }
           break;
         };
    break;

    case BVH_ROTATION_Y :
         switch (typeB)
         {
           case BVH_ROTATION_X :
               if (typeC == BVH_ROTATION_Z) { return BVH_ROTATION_ORDER_YXZ; }
           break;
           case BVH_ROTATION_Z :
               if (typeC == BVH_ROTATION_X) { return BVH_ROTATION_ORDER_YZX; }
           break;
         };
    break;

    case BVH_ROTATION_Z :
         switch (typeB)
         {
           case BVH_ROTATION_X :
               if (typeC == BVH_ROTATION_Y) { return BVH_ROTATION_ORDER_ZXY; }
           break;
           case BVH_ROTATION_Y :
               if (typeC == BVH_ROTATION_X) { return BVH_ROTATION_ORDER_ZYX; }
           break;
         };
    break;
  }
 return BVH_ROTATION_ORDER_NONE;
}


int enumerateChannelOrder(struct BVH_MotionCapture * bvhMotion , unsigned int currentJoint)
{
  int channelOrder=enumerateChannelOrderFromTypes(
                                                  bvhMotion->jointHierarchy[currentJoint].channelType[0],
                                                  bvhMotion->jointHierarchy[currentJoint].channelType[1],
                                                  bvhMotion->jointHierarchy[currentJoint].channelType[2]
                                                 );

  if (channelOrder==BVH_ROTATION_ORDER_NONE)
  {
      channelOrder=enumerateChannelOrderFromTypes(
                                                  bvhMotion->jointHierarchy[currentJoint].channelType[3],
                                                  bvhMotion->jointHierarchy[currentJoint].channelType[4],
                                                  bvhMotion->jointHierarchy[currentJoint].channelType[5]
                                                 );
  }
  if (channelOrder==BVH_ROTATION_ORDER_NONE)
  {
    fprintf(stderr,RED "BUG: Channel order still wrong, todo smarter channel order enumeration..\n" NORMAL);
  }
return channelOrder;
}


unsigned int bvh_resolveFrameAndJointAndChannelToMotionID(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID)
{
   if ( (channelTypeID<BVH_VALID_CHANNEL_NAMES) && (jID<bvhMotion->jointHierarchySize) )
   {
     return  (fID * bvhMotion->numberOfValuesPerFrame) + bvhMotion->jointToMotionLookup[jID].channelIDMotionOffset[channelTypeID]; 
   }

  return 0; 
}




//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        MOTION PARSING
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------



//----------------------------------------------------------------------------------------------------
int bvh_loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion, float scaleWorld)
{
 //fprintf(stderr,"Loading BVH %s \n",filename);
 bvhMotion->scaleWorld=scaleWorld;
  int successfullRead=0;
  FILE *fd=0;
  fd = fopen(filename,"r");
  if (fd!=0)
    {
      snprintf(bvhMotion->fileName,2047,"%s",filename);
      if (readBVHHeader(bvhMotion,fd))
      {
       //If we have the header let's update the hashes
       bvh_updateJointNameHashes(bvhMotion);

       if (readBVHMotion(bvhMotion,fd))
       {
         successfullRead=1;
       }
      }
      fclose(fd);
    }
 return successfullRead;
}


int bvh_free(struct BVH_MotionCapture * bvhMotion)
{
  if (bvhMotion==0) { return 0; }
  if (bvhMotion->motionValues!=0)               {  free(bvhMotion->motionValues);       bvhMotion->motionValues=0;  }
  if (bvhMotion->selectedJoints!=0)             {  free(bvhMotion->selectedJoints);     bvhMotion->selectedJoints=0;  }
  if (bvhMotion->hideSelectedJoints!=0)         {  free(bvhMotion->hideSelectedJoints); bvhMotion->hideSelectedJoints=0;  }

  memset(bvhMotion,0,sizeof(struct BVH_MotionCapture));

  return 1;
}
//----------------------------------------------------------------------------------------------------




int bvh_SetPositionRotation(
                             struct BVH_MotionCapture * mc,
                             float * position,
                             float * rotation
                            )
{
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;
   mc->motionValues[mID+0]=position[0];
   mc->motionValues[mID+1]=position[1];
   mc->motionValues[mID+2]=position[2];
   mc->motionValues[mID+3]=rotation[0];
   mc->motionValues[mID+4]=rotation[1];
   mc->motionValues[mID+5]=rotation[2];
  }
 return 1;
}


int bvh_OffsetPositionRotation(
                               struct BVH_MotionCapture * mc,
                               float * position,
                               float * rotation
                              )
{
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;
   mc->motionValues[mID+0]+=position[0];
   mc->motionValues[mID+1]+=position[1];
   mc->motionValues[mID+2]+=position[2];
   mc->motionValues[mID+3]+=rotation[0];
   mc->motionValues[mID+4]+=rotation[1];
   mc->motionValues[mID+5]+=rotation[2];
  }
 return 1;
}



int bvh_ConstrainRotations(
                           struct BVH_MotionCapture * mc,
                           unsigned int constrainOrientation
                          )
{
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;

   float buffer = (float) mc->motionValues[mID+3];
   buffer = bvh_RemapAngleCentered0(buffer,0);
   mc->motionValues[mID+3] = (float) buffer;

   buffer = (float) mc->motionValues[mID+4];
   buffer = bvh_RemapAngleCentered0(buffer,constrainOrientation);
   mc->motionValues[mID+4] = (float) buffer;

   buffer = (float) mc->motionValues[mID+5];
   buffer = bvh_RemapAngleCentered0(buffer,0);
   mc->motionValues[mID+5] = (float) buffer;
  }
 return 1;
}







//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        ACCESSORS
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
int bhv_jointHasParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID )
{
 if (jID<bvhMotion->jointHierarchySize) 
     { 
       return (!bvhMotion->jointHierarchy[jID].isRoot);
     }
 return 0;
}

int bhv_jointGetEndSiteChild(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,BVHJointID * jChildID)
{
  //fprintf(stderr," bhv_jointGetEndSiteChild ");
  if (bvhMotion==0) { return 0; }

  if (bvhMotion->jointHierarchy[jID].hasEndSite)
  {
   unsigned int jointID=0;
   for (jointID=0; jointID<bvhMotion->jointHierarchySize; jointID++)
   {
     //fprintf(stderr,"joint[%u]=%s ",jointID,bvhMotion->jointHierarchy[jointID].jointName);
     if (bvhMotion->jointHierarchy[jointID].isEndSite)
     {
          if (bvhMotion->jointHierarchy[jointID].parentJoint==jID)
          {
              *jChildID=jointID;
              return 1;
          }
     }
   }
  }
 return 0;
}


int bhv_jointHasRotation(struct BVH_MotionCapture * bvhMotion , BVHJointID jID)
{
 if (jID>bvhMotion->jointHierarchySize) { return 0; }
 return (
          (bvhMotion->jointHierarchy[jID].loadedChannels>0) &&
          (bvhMotion->jointHierarchy[jID].channelRotationOrder!=0)
        );
}



int bvh_getJointIDFromJointName(
                                 struct BVH_MotionCapture * bvhMotion ,
                                 const char * jointName,
                                 BVHJointID * jID
                                )
{
   if (bvhMotion==0) { return 0; }
   if (jointName==0) { return 0; }
   if (jID==0)       { return 0; }
   unsigned int i=0;
   for (i=0; i<bvhMotion->jointHierarchySize; i++)
   {
     if (strcmp(bvhMotion->jointHierarchy[i].jointName,jointName)==0)
     {
         *jID=i;
         return 1;
     }
   }
 return 0;
}


int bvh_getJointIDFromJointNameNocase(
                                      struct BVH_MotionCapture * bvhMotion ,
                                      const char * jointName,
                                      BVHJointID * jID
                                     )
{
  if (bvhMotion==0) { return 0; }
  if (jointName==0) { return 0; }
  if (jID==0)       { return 0; }
  
  if (strlen(jointName)>=MAX_BVH_JOINT_NAME)
     {
       fprintf(stderr,"bvh_getJointIDFromJointNameNocase failed because of very long joint names..");
       return 0;
     }

   char jointNameLowercase[MAX_BVH_JOINT_NAME+1]={0};
   snprintf(jointNameLowercase,MAX_BVH_JOINT_NAME,"%s",jointName);
   lowercase(jointNameLowercase);

  unsigned int i=0;
   for (i=0; i<bvhMotion->jointHierarchySize; i++)
   {
     if (strcmp(bvhMotion->jointHierarchy[i].jointNameLowercase,jointNameLowercase)==0)
     {
         *jID=i;
         return 1;
     }
   }
 return 0;
}







int bvh_getRootJointID(
                       struct BVH_MotionCapture * bvhMotion ,
                       BVHJointID * jID
                      )
{
  if (bvhMotion==0) { return 0; }
  if (jID==0)       { return 0; }

  *jID = bvhMotion->rootJointID;
  return 1;
  /*
   unsigned int i=0;
   for (i=0; i<bvhMotion->jointHierarchySize; i++)
   {
     if (bvhMotion->jointHierarchy[i].isRoot)
     {
         *jID=i;
         return 1;
     }
   }
 return 0;
 */
}


int bhv_getJointParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID)
{
   if (bvhMotion==0) { return 0; }
   if (bvhMotion->jointHierarchySize>jID)
     {
       return bvhMotion->jointHierarchy[jID].parentJoint;
     }
   return 0;
}

int bvh_onlyAnimateGivenJoints(struct BVH_MotionCapture * bvhMotion,unsigned int numberOfArguments,const char **argv)
{
    bvh_printBVH(bvhMotion);
    fprintf(stderr,"bvh_onlyAnimateGivenJoints with %u arguments\n",numberOfArguments);

    BVHJointID * activeJoints = (BVHJointID*) malloc(sizeof(BVHJointID) * numberOfArguments);
    //-----------------------------------------------------------------------------------------------
    if (activeJoints==0)
    {
      fprintf(stderr,"bvh_onlyAnimateGivenJoints failed to allocate space for %u arguments\n",numberOfArguments);
      return 0;
    } else
    {
      memset(activeJoints,0,sizeof(BVHJointID) * numberOfArguments);
    }



    char * successJoints = (char *) malloc(sizeof(char) * numberOfArguments);
    //-----------------------------------------------------------------------------------------------
    if (successJoints==0)
    {
      fprintf(stderr,"bvh_onlyAnimateGivenJoints failed to allocate space for %u arguments\n",numberOfArguments);
      free(activeJoints);
      return 0;
    } else
    {
     memset(successJoints,0,sizeof(char) * numberOfArguments);
    }

    if ((activeJoints!=0) && (successJoints!=0))
    {

    for (unsigned int i=0; i<numberOfArguments; i++)
    {
      BVHJointID jID=0;

      if (
           bvh_getJointIDFromJointNameNocase(
                                             bvhMotion ,
                                             argv[i],
                                             &jID
                                            )
         )
         {
           fprintf(stderr,GREEN "Joint Activated %u = %s -> jID=%u\n" NORMAL,i,argv[i],jID);
           activeJoints[i]=jID;
           successJoints[i]=1;
         } else
         {
           fprintf(stderr,RED "Joint Failed to Activate %u = %s\n" NORMAL,i,argv[i]);
           fprintf(stderr,RED "Check the list above to find correct joint names..\n" NORMAL);
         }
    }


      unsigned int mID_Initial,mID_Target;
      for (int frameID=0; frameID<bvhMotion->numberOfFramesEncountered; frameID++)
       {
         //fprintf(stderr,"FrameNumber %u\n",frameID);
         for (int mID=0; mID<bvhMotion->numberOfValuesPerFrame; mID++)
         {
             unsigned int  jointID = bvhMotion->motionToJointLookup[mID].jointID;
             int isMIDProtected=0;

             for (int aJ=0; aJ<numberOfArguments; aJ++)
              {
               if (successJoints[aJ])
                {
                   if (jointID==activeJoints[aJ])
                   {
                     isMIDProtected=1;
                   }
                }
              }

            if (!isMIDProtected)
            {
              mID_Initial=mID;
              mID_Target=frameID * bvhMotion->numberOfValuesPerFrame + mID;
              bvhMotion->motionValues[mID_Target] = bvhMotion->motionValues[mID_Initial];
            }

         }

         /* This does the inverse
         unsigned int firstFrame=0;
         for (int aJ=0; aJ<numberOfArguments; aJ++)
         {
           if (successJoints[aJ])
           {
            jointID = activeJoints[aJ];

            mID_Initial = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,firstFrame,BVH_ROTATION_X);
            mID_Target = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,frameID,BVH_ROTATION_X);
            bvhMotion->motionValues[mID_Target] = bvhMotion->motionValues[mID_Initial];

            mID_Initial = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,firstFrame,BVH_ROTATION_Y);
            mID_Target = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,frameID,BVH_ROTATION_Y);
            bvhMotion->motionValues[mID_Target] = bvhMotion->motionValues[mID_Initial];

            mID_Initial = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,firstFrame,BVH_ROTATION_Z);
            mID_Target = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jointID,frameID,BVH_ROTATION_Z);
            bvhMotion->motionValues[mID_Target] = bvhMotion->motionValues[mID_Initial];
           }
         }
         */
       }
      free(successJoints);
      free(activeJoints);
      return 1;
    }

  return 0;
}


int bvh_changeJointDimensions(
                              struct BVH_MotionCapture * bvhMotion,
                              const char * jointName,
                              float xScale,
                              float yScale,
                              float zScale
                             )
{
   if (bvhMotion==0)
    {
     fprintf(stderr,"bvh_changeJointDimensions: Cannot work before having loaded a bvh file\n");
     return 0;
    }




    //fprintf(stderr,"bvh_changeJointDimensions: %s %0.2f %0.2f %0.2f\n",jointName,xScale,yScale,zScale);

    BVHJointID jID=0;
   if ( bvh_getJointIDFromJointNameNocase(bvhMotion,jointName,&jID) )
   //if ( bvh_getJointIDFromJointName(bvhMotion ,jointName,&jID) )
   {
       unsigned int angleX = 0;
       unsigned int angleY = 1;
       unsigned int angleZ = 2;

       for (int ch=0; ch<3; ch++)
       {
        if (bvhMotion->jointHierarchy[jID].channelType[ch]==BVH_ROTATION_X) { angleX = ch; }
        if (bvhMotion->jointHierarchy[jID].channelType[ch]==BVH_ROTATION_Y) { angleY = ch; }
        if (bvhMotion->jointHierarchy[jID].channelType[ch]==BVH_ROTATION_Z) { angleZ = ch; }
       }

       //fprintf(stderr,"offset was %0.2f %0.2f %0.2f\n",bvhMotion->jointHierarchy[jID].offset[0],bvhMotion->jointHierarchy[jID].offset[1],bvhMotion->jointHierarchy[jID].offset[2]);
       bvhMotion->jointHierarchy[jID].offset[angleX] *= xScale;
       bvhMotion->jointHierarchy[jID].offset[angleY] *= yScale;
       bvhMotion->jointHierarchy[jID].offset[angleZ] *= zScale;
       //fprintf(stderr,"offset is now %0.2f %0.2f %0.2f\n",bvhMotion->jointHierarchy[jID].offset[0],bvhMotion->jointHierarchy[jID].offset[1],bvhMotion->jointHierarchy[jID].offset[2]);

       float * m = bvhMotion->jointHierarchy[jID].staticTransformation.m;
       m[0] =1.0;  m[1] =0.0;  m[2] =0.0;  m[3] = (float) bvhMotion->jointHierarchy[jID].offset[0];
       m[4] =0.0;  m[5] =1.0;  m[6] =0.0;  m[7] = (float) bvhMotion->jointHierarchy[jID].offset[1];
       m[8] =0.0;  m[9] =0.0;  m[10]=1.0;  m[11]= (float) bvhMotion->jointHierarchy[jID].offset[2];
       m[12]=0.0;  m[13]=0.0;  m[14]=0.0;  m[15]=1.0;

      return 1;
   }

 fprintf(stderr,"bvh_changeJointDimensions: Unable to locate joint %s \n",jointName);
 fprintf(stderr,"Joint List : ");
 for (unsigned int jIDIt=0; jIDIt<bvhMotion->jointHierarchySize; jIDIt++)
 {
   if (jIDIt!=0) { fprintf(stderr,","); }
   fprintf(stderr,"%s(or. %s)",bvhMotion->jointHierarchy[jIDIt].jointNameLowercase,bvhMotion->jointHierarchy[jIDIt].jointName);
 }
 fprintf(stderr,"\n");
 return 0;
}



int bvh_scaleAllOffsets(
                        struct BVH_MotionCapture * bvhMotion,
                        float scalingRatio
                       )
{
   if (bvhMotion==0) { return 0; }
   if (scalingRatio==1.0) { return 1; }

    for (BVHJointID jID=0; jID<bvhMotion->jointHierarchySize; jID++)
    {
      bvhMotion->jointHierarchy[jID].offset[0] = bvhMotion->jointHierarchy[jID].offset[0] * scalingRatio;
      bvhMotion->jointHierarchy[jID].offset[1] = bvhMotion->jointHierarchy[jID].offset[1] * scalingRatio;
      bvhMotion->jointHierarchy[jID].offset[2] = bvhMotion->jointHierarchy[jID].offset[2] * scalingRatio;

       float * m = bvhMotion->jointHierarchy[jID].staticTransformation.m;
       m[0] =1.0;  m[1] =0.0;  m[2] =0.0;  m[3] = (float) bvhMotion->jointHierarchy[jID].offset[0];
       m[4] =0.0;  m[5] =1.0;  m[6] =0.0;  m[7] = (float) bvhMotion->jointHierarchy[jID].offset[1];
       m[8] =0.0;  m[9] =0.0;  m[10]=1.0;  m[11]= (float) bvhMotion->jointHierarchy[jID].offset[2];
       m[12]=0.0;  m[13]=0.0;  m[14]=0.0;  m[15]=1.0;
    }

 return 1;
}




int bvh_getMotionChannelName(struct BVH_MotionCapture * bvhMotion,BVHMotionChannelID mID,char * target,unsigned int targetLength)
{
 if (mID<bvhMotion->numberOfValuesPerFrame)
 {
   BVHJointID jID = bvhMotion->motionToJointLookup[mID].jointID;
   unsigned int channelID = bvhMotion->motionToJointLookup[mID].channelID;
   if ( (jID<bvhMotion->jointHierarchySize) && (channelID<BVH_VALID_CHANNEL_NAMES) )
   {
      char * jointLabel = bvhMotion->jointHierarchy[jID].jointName;

      int i = snprintf(target,targetLength,"%s_%s",jointLabel,channelNames[channelID]);
      if (i<0)
      {
         fprintf(stderr,"Not enough space to hold motion channel name for mID=%u\n",mID);
         return 0;
      }
   }
 }
 return 0;
}








//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
float bvh_getJointChannelAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID)
{
   if ( (bvhMotion!=0) && (jID<bvhMotion->jointHierarchySize) ) 
    { 
       
      unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,fID,channelTypeID);

      if (mID<bvhMotion->motionValuesSize)
       {
         return bvhMotion->motionValues[mID];
       } else
       { 
         fprintf(stderr,RED "bvh_getJointChannelAtFrame overflowed..\n" NORMAL);
       } 
}
 return 0.0;
}

float  bvh_getJointRotationXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_X); }
float  bvh_getJointRotationYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Y); }
float  bvh_getJointRotationZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Z); }
float  bvh_getJointPositionXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_X); }
float  bvh_getJointPositionYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Y); }
float  bvh_getJointPositionZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID) { return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Z); }

int bhv_populatePosXYZRotXYZ(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID , float * data , unsigned int sizeOfData)
{
  if (data == 0) { return 0; }
  if (sizeOfData < sizeof(float)* 6) { return 0; }

  data[0]=bvh_getJointPositionXAtFrame(bvhMotion,jID,fID);
  data[1]=bvh_getJointPositionYAtFrame(bvhMotion,jID,fID);
  data[2]=bvh_getJointPositionZAtFrame(bvhMotion,jID,fID);
  data[3]=bvh_getJointRotationXAtFrame(bvhMotion,jID,fID);
  data[4]=bvh_getJointRotationYAtFrame(bvhMotion,jID,fID);
  data[5]=bvh_getJointRotationZAtFrame(bvhMotion,jID,fID);
  return 1;
}
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------



//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
int bvh_setJointChannelAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID,float value)
{
   if (bvhMotion==0) { return 0.0; }
   if (bvhMotion->jointHierarchySize<=jID) { return 0.0; }

   unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,fID,channelTypeID);

   if (mID>=bvhMotion->motionValuesSize)
   {
     fprintf(stderr,RED "bvh_setJointChannelAtFrame overflowed..\n" NORMAL);
     return 0;
   }

    bvhMotion->motionValues[mID]=value;

    return 1;
}

int bvh_setJointRotationXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_X,value); }
int bvh_setJointRotationYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Y,value); }
int bvh_setJointRotationZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Z,value); }
int bvh_setJointPositionXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_X,value); }
int bvh_setJointPositionYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Y,value); }
int bvh_setJointPositionZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID,float value) { return bvh_setJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Z,value); }


int bhv_setPosXYZRotXYZ(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID , float * data , unsigned int sizeOfData)
{
  if (data == 0) { return 0; }
  if (sizeOfData < sizeof(float)* 6) { return 0; }

  int successfulStores=0;
  successfulStores+=bvh_setJointPositionXAtFrame(bvhMotion,jID,fID,data[0]);
  successfulStores+=bvh_setJointPositionYAtFrame(bvhMotion,jID,fID,data[1]);
  successfulStores+=bvh_setJointPositionZAtFrame(bvhMotion,jID,fID,data[2]);
  successfulStores+=bvh_setJointRotationXAtFrame(bvhMotion,jID,fID,data[3]);
  successfulStores+=bvh_setJointRotationYAtFrame(bvhMotion,jID,fID,data[4]);
  successfulStores+=bvh_setJointRotationZAtFrame(bvhMotion,jID,fID,data[5]);
  return (successfulStores==6);
}

//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
float bvh_getJointChannelAtMotionBuffer(struct BVH_MotionCapture * bvhMotion, BVHJointID jID,float * motionBuffer, unsigned int channelTypeID)
{
   if ( (bvhMotion!=0) && (jID<bvhMotion->jointHierarchySize) ) 
       { 
         unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,0,channelTypeID);

         if (mID<bvhMotion->motionValuesSize)
           {
             return motionBuffer[mID];
           }
         fprintf(stderr,RED "bvh_getJointChannelAtMotionBuffer overflowed..\n" NORMAL);
       }
 return 0.0; 
}

float  bvh_getJointRotationXAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_X); }
float  bvh_getJointRotationYAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_Y); }
float  bvh_getJointRotationZAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_Z); }
float  bvh_getJointPositionXAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_POSITION_X); }
float  bvh_getJointPositionYAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_POSITION_Y); }
float  bvh_getJointPositionZAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_POSITION_Z); }


int bhv_populatePosXYZRotXYZFromMotionBuffer(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , float * motionBuffer, float * data, unsigned int sizeOfData)
{
  //This gets spammed a *LOT* so it needs to be improved..
  if ( (motionBuffer!=0) && (data!=0) && (sizeOfData >= sizeof(float)* 6) ) 
  {
     // Old/Clean implementation..
      if (!bvhMotion->jointHierarchy[jID].isRoot)
      {
       data[0]=0.0;
       data[1]=0.0;
       data[2]=0.0;         
      } else
      {//Only Root joint has a position field..
       data[0]=bvh_getJointPositionXAtMotionBuffer(bvhMotion,jID,motionBuffer);
       data[1]=bvh_getJointPositionYAtMotionBuffer(bvhMotion,jID,motionBuffer);
       data[2]=bvh_getJointPositionZAtMotionBuffer(bvhMotion,jID,motionBuffer);  
      }
      
      
      
       //This code segment used to be just these 6 simple lines, however it run 2x slower :P
       //data[0]=bvh_getJointPositionXAtMotionBuffer(bvhMotion,jID,motionBuffer);
       //data[1]=bvh_getJointPositionYAtMotionBuffer(bvhMotion,jID,motionBuffer);
       //data[2]=bvh_getJointPositionZAtMotionBuffer(bvhMotion,jID,motionBuffer);
       //data[3]=bvh_getJointRotationXAtMotionBuffer(bvhMotion,jID,motionBuffer);
       //data[4]=bvh_getJointRotationYAtMotionBuffer(bvhMotion,jID,motionBuffer);
       //data[5]=bvh_getJointRotationZAtMotionBuffer(bvhMotion,jID,motionBuffer);
       
       unsigned int mID;
       
       switch (bvhMotion->jointHierarchy[jID].channelRotationOrder)
       {
           case BVH_ROTATION_ORDER_ZXY : 
             //Special code to speed up cases that match the BVH specification ZXY rotation orders
             mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,0,BVH_ROTATION_X);
             data[3]=motionBuffer[mID];
             mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,0,BVH_ROTATION_Y);
             data[4]=motionBuffer[mID];
             mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,0,BVH_ROTATION_Z);
             data[5]=motionBuffer[mID];
           break;
           
           default : 
             data[3]=bvh_getJointRotationXAtMotionBuffer(bvhMotion,jID,motionBuffer);
             data[4]=bvh_getJointRotationYAtMotionBuffer(bvhMotion,jID,motionBuffer);
             data[5]=bvh_getJointRotationZAtMotionBuffer(bvhMotion,jID,motionBuffer);
           break;
       };
       
   
    return 1;
  }
  
  return 0;  
}
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------



float bvh_getMotionValue(struct BVH_MotionCapture * bvhMotion , unsigned int mID)
{
  return bvhMotion->motionValues[mID];
}



int bvh_copyMotionFrameToMotionBuffer(
                                       struct BVH_MotionCapture * bvhMotion,
                                       struct MotionBuffer * motionBuffer,
                                       BVHFrameID fromfID
                                     )
{
   if (motionBuffer==0) { return 0; }
   if (motionBuffer->motion==0) { return 0; }
   if (motionBuffer->bufferSize < bvhMotion->numberOfValuesPerFrame) { return 0; }

   if ( fromfID < bvhMotion->numberOfFrames )
   {
     memcpy(
             motionBuffer->motion,
             &bvhMotion->motionValues[fromfID * bvhMotion->numberOfValuesPerFrame],
             bvhMotion->numberOfValuesPerFrame * sizeof(float)
           );
     return 1;
   }
 return 0;
}







int bvh_copyMotionFrame(
                         struct BVH_MotionCapture * bvhMotion,
                         BVHFrameID tofID,
                         BVHFrameID fromfID
                        )
{
   if (
         (tofID<bvhMotion->numberOfFrames ) && (fromfID<bvhMotion->numberOfFrames )
      )
   {
     memcpy(
             &bvhMotion->motionValues[tofID * bvhMotion->numberOfValuesPerFrame],
             &bvhMotion->motionValues[fromfID * bvhMotion->numberOfValuesPerFrame],
             bvhMotion->numberOfValuesPerFrame * sizeof(float)
           );
     return 1;
   }
 return 0;
}


int bvh_selectJoints(
                    struct BVH_MotionCapture * mc,
                    unsigned int numberOfValues,
                    unsigned int includeEndSites,
                    const char **argv,
                    unsigned int iplus1
                   )
{
  fprintf(stderr,"Asked to select %u Joints\n",numberOfValues);

  mc->selectionIncludesEndSites=includeEndSites;
  mc->numberOfJointsWeWantToSelect=numberOfValues;

  //Uncomment to force each call to selectJoints to invalidade previous calls..
  //if (mc->selectedJoints!=0) { free(mc->selectedJoints); mc->selectedJoints=0; }
  if (mc->selectedJoints!=0) {
                               fprintf(stderr,YELLOW "Multiple selection of joints taking place..\n" NORMAL);
                             } else
                             {
                              mc->selectedJoints = (unsigned int *) malloc(sizeof(unsigned int) * mc->numberOfValuesPerFrame);
                              if (mc->selectedJoints==0)
                                   {
                                     fprintf(stderr,RED "bvh_selectJoints failed to allocate selectedJoints\n" NORMAL);
                                     return 0;
                                   } else
                                   {
                                      memset(mc->selectedJoints,0,sizeof(unsigned int)* mc->numberOfValuesPerFrame);
                                   }
                             }


  //if (mc->hideSelectedJoints!=0) { free(mc->hideSelectedJoints); mc->hideSelectedJoints=0; }
   if (mc->hideSelectedJoints!=0) {
                                    fprintf(stderr,YELLOW "Multiple selection of hidden joints taking place..\n" NORMAL);
                                  } else
                                  {
                                     mc->hideSelectedJoints = (unsigned int *) malloc(sizeof(unsigned int) * mc->numberOfValuesPerFrame);
                                     if (mc->hideSelectedJoints==0)
                                          {
                                            fprintf(stderr,RED "bvh_selectJoints failed to allocate hideSelectedJoints\n" NORMAL);
                                            free(mc->selectedJoints);
                                            mc->selectedJoints=0;
                                            return 0;
                                          } else
                                          {
                                             memset(mc->hideSelectedJoints,0,sizeof(unsigned int)* mc->numberOfValuesPerFrame);
                                          }

                                  }




  if ( (mc->selectedJoints!=0) && (mc->hideSelectedJoints!=0) )
  {
    unsigned int success=1;

    BVHJointID jID=0;
    fprintf(stderr,"Selecting : ");

    unsigned int i=0;
    for (i=iplus1+1; i<=iplus1+numberOfValues; i++)
     {
        if (
              //bvh_getJointIDFromJointName(mc,argv[i],&jID)
              bvh_getJointIDFromJointNameNocase(mc,argv[i],&jID)
           )
         {
           fprintf(stderr,GREEN "%s " NORMAL,argv[i]);

           mc->selectedJoints[jID]=1;
           mc->hideSelectedJoints[jID]=0;
           fprintf(stderr,"%u ",jID);

           if(includeEndSites)
                   {
                       if (mc->jointHierarchy[jID].hasEndSite)
                       {
                            fprintf(stderr,GREEN "EndSite_%s  " NORMAL,argv[i]);
                       }
                   }
           //-------------------------------------------------

         } else
         {
           fprintf(stderr,RED "%s(not found) " NORMAL,argv[i]);
           success=0;
         }
     }
    fprintf(stderr,"\n");


    return success;
  }

  //We failed to allocate both so deallocate everything..
  mc->selectionIncludesEndSites=0;
  mc->numberOfJointsWeWantToSelect=0;
  if (mc->selectedJoints!=0) { free(mc->selectedJoints); mc->selectedJoints=0; }
  if (mc->hideSelectedJoints!=0) { free(mc->hideSelectedJoints); mc->hideSelectedJoints=0; }

  return 0;
}


int bvh_selectJointsToHide2D(
                             struct BVH_MotionCapture * mc,
                             unsigned int numberOfValues,
                             unsigned int includeEndSites,
                             const char **argv,
                             unsigned int iplus1
                            )
{
  if ( (mc->selectedJoints!=0) && (mc->hideSelectedJoints!=0) )
  {
    unsigned int success=1;

    unsigned int i=0;
    BVHJointID jID=0;
    fprintf(stderr,"Hiding 2D Coordinates : ");
    for (i=iplus1+1; i<=iplus1+numberOfValues; i++)
     {
      if (
              bvh_getJointIDFromJointNameNocase(mc,argv[i],&jID)
              //bvh_getJointIDFromJointName(mc,argv[i],&jID)
           )
         {
           fprintf(stderr,GREEN "%s " NORMAL,argv[i]);
           mc->hideSelectedJoints[jID]=1;
           fprintf(stderr,"%u ",jID);

           if(includeEndSites)
                   {
                     if(mc->selectionIncludesEndSites)
                          {
                              if (mc->jointHierarchy[jID].hasEndSite)
                                {
                                  fprintf(stderr,GREEN "EndSite_%s  " NORMAL,argv[i]);
                                }
                          }
                   } else
                   {
                    if (mc->jointHierarchy[jID].hasEndSite)
                                {
                                  fprintf(stderr,YELLOW "EndSite_%s(protected) " NORMAL,argv[i]);
                                  mc->hideSelectedJoints[jID]=2;
                                }
                   }
           //-------------------------------------------------

         } else
         {
           fprintf(stderr,RED "%s(not found) " NORMAL,argv[i]);
           success=0;
         }
     }
    fprintf(stderr,"\n");
    return success;
  }

 fprintf(stderr,RED "bvh_selectJointsToHide2D cannot work without a prior bvh_selectJoints call..\n" NORMAL);
 return 0;
}











//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        PRINT STATE
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------





 int bvh_removeSelectedFrames(struct BVH_MotionCapture * bvhMotion,unsigned int * framesToRemove)
 {
   if ( (bvhMotion==0)||(framesToRemove==0) ) { return 0; }

   unsigned int offsetMID=0;
   unsigned int copyingEngaged=0;
   unsigned int framesEliminated=0;
   //---------------------------------------------------
   for (int fID=0; fID<bvhMotion->numberOfFrames; fID++)
   {
     if (framesToRemove[fID])
     {
        copyingEngaged=1;
        offsetMID+=bvhMotion->numberOfValuesPerFrame;
        ++framesEliminated;
     }

     if (copyingEngaged)
     {
       unsigned int mIDStart = fID * bvhMotion->numberOfValuesPerFrame;
       unsigned int mIDEnd   = (fID+1) * bvhMotion->numberOfValuesPerFrame;

       if (mIDEnd+offsetMID>=bvhMotion->numberOfFrames * bvhMotion->numberOfValuesPerFrame)
       {
          //Done erasing..
          break;
          //--------------
       }  else
       {
        for (unsigned int mID=mIDStart; mID<mIDEnd; mID++)
          {
            bvhMotion->motionValues[mID]=bvhMotion->motionValues[mID+offsetMID];
          }
       }
     }
   }
   //---------------------------------------------------

   bvhMotion->numberOfFramesEncountered=bvhMotion->numberOfFramesEncountered-framesEliminated;
   bvhMotion->numberOfFrames=bvhMotion->numberOfFrames-framesEliminated;

   return 1;
 }










//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        MOTION BUFFER
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------

void freeMotionBuffer(struct MotionBuffer ** pointerToMB)
{
 struct MotionBuffer *mb = *pointerToMB;
 if (mb!=0)
 {
  if(mb->motion!=0)
   {
     free(mb->motion);
     mb->motion=0;
   }
  free(mb);
  //Also clean the pointer..
  *pointerToMB=0;
 }
}
//---------------------------------------------------------

int copyMotionBuffer(struct MotionBuffer * dst,struct MotionBuffer * src)
{
  if ( (src!=0) && (dst!=0) )
  {
   if (src->bufferSize == dst->bufferSize)
   {
     // 0.01  
     memcpy(dst->motion,src->motion,sizeof(float) * src->bufferSize);
     // 0.16
     /*for (unsigned int i=0; i<dst->bufferSize; i++)
      { dst->motion[i] = src->motion[i]; } */
     return 1;
   }
    else
   {
    fprintf(stderr,RED "copyMotionBuffer: Buffer Size mismatch (Source %u/Destination %u)..\n" NORMAL,src->bufferSize,dst->bufferSize);
    return 0;
   }
  
  }   
  
  return 0;
}

struct MotionBuffer * mallocNewMotionBuffer(struct BVH_MotionCapture * mc)
{
  if (mc==0) { return 0; }
  //----------------------
  
  struct MotionBuffer * newBuffer = (struct MotionBuffer *)  malloc(sizeof(struct MotionBuffer));
  if (newBuffer!=0)
  {
    newBuffer->bufferSize = mc->numberOfValuesPerFrame;
    newBuffer->motion = (float *) malloc(sizeof(float) * newBuffer->bufferSize);
    if (newBuffer->motion!=0)
    {
      memset(newBuffer->motion,0,sizeof(float) * newBuffer->bufferSize);
    } else
    {
      //RollBack..!  
      newBuffer->bufferSize=0;
      free(newBuffer);
      newBuffer=0;
    }
  }  
  //----------------------
   return newBuffer;
}
//---------------------------------------------------------


struct MotionBuffer * mallocNewMotionBufferAndCopy(struct BVH_MotionCapture * mc,struct MotionBuffer * whatToCopy)
{
  if (mc==0) { return 0; }
  if (whatToCopy==0) { fprintf(stderr,"mallocNewMotionBufferAndCopy: Unable to copy from empty"); return 0; }
  if (whatToCopy->motion==0) { fprintf(stderr,"mallocNewMotionBufferAndCopy: Unable to copy from empty"); return 0; }
  //----------------------

  struct MotionBuffer * newBuffer = (struct MotionBuffer *)  malloc(sizeof(struct MotionBuffer));
  if (newBuffer!=0)
  {
    newBuffer->bufferSize = mc->numberOfValuesPerFrame;
    if (mc->numberOfValuesPerFrame != whatToCopy->bufferSize)
    {
        fprintf(stderr,RED "mallocNewMotionBufferAndCopy: Mismatching sizes %u vs %u\n" NORMAL,mc->numberOfValuesPerFrame,whatToCopy->bufferSize); 
    }
      
    newBuffer->motion = (float *) malloc(sizeof(float) * newBuffer->bufferSize);
    if (newBuffer->motion!=0)
    {
      unsigned int numberOfItems = newBuffer->bufferSize; 
      if (mc->numberOfValuesPerFrame>whatToCopy->bufferSize)  
           { numberOfItems = newBuffer->bufferSize; }
        
      for (unsigned int i=0; i<numberOfItems; i++)
      {
        newBuffer->motion[i]=whatToCopy->motion[i];
      }
    } else
    {
      //RollBack..!  
      newBuffer->bufferSize=0; 
      free(newBuffer);
      newBuffer=0;
    }
  }

  //----------------------
  return newBuffer;
}
//---------------------------------------------------------



void compareMotionBuffers(const char * msg,struct MotionBuffer * guess,struct MotionBuffer * groundTruth)
{
  fprintf(stderr,"%s \n",msg);
  fprintf(stderr,"___________\n");

  if (guess->bufferSize != groundTruth->bufferSize)
  {
    fprintf(stderr,"compareMotionBuffers: Buffer Size mismatch..\n");
    return ;
  }

  //--------------------------------------------------
  fprintf(stderr,"Guess : ");
  for (unsigned int i=0; i<guess->bufferSize; i++)
  {
    fprintf(stderr,"%0.2f " ,guess->motion[i]);
  }
  fprintf(stderr,"\n");
  //--------------------------------------------------
  fprintf(stderr,"Truth : ");
  for (unsigned int i=0; i<groundTruth->bufferSize; i++)
  {
    fprintf(stderr,"%0.2f " ,groundTruth->motion[i]);
  }
  fprintf(stderr,"\n");
  //--------------------------------------------------


  fprintf(stderr,"Diff : ");

  for (unsigned int i=0; i<guess->bufferSize; i++)
  {
    float diff=fabs(groundTruth->motion[i] - guess->motion[i]);
    if (fabs(diff)<0.1) { fprintf(stderr,GREEN "%0.2f " ,diff); } else
                         { fprintf(stderr,RED "%0.2f " ,diff); }
  }
  fprintf(stderr,NORMAL "\n___________\n");
}


void compareTwoMotionBuffers(struct BVH_MotionCapture * mc,const char * msg,struct MotionBuffer * guessA,struct MotionBuffer * guessB,struct MotionBuffer * groundTruth)
{
  fprintf(stderr,"%s \n",msg);
  fprintf(stderr,"___________\n");

  if ( (guessA->bufferSize != groundTruth->bufferSize) || (guessB->bufferSize != groundTruth->bufferSize) )
  {
    fprintf(stderr,"compareTwoMotionBuffers: Buffer Size mismatch..\n");
    return ;
  }


  fprintf(stderr,"Diff : ");
  for (unsigned int i=0; i<guessA->bufferSize; i++)
  {
    float diffA=fabs(groundTruth->motion[i] - guessA->motion[i]);
    float diffB=fabs(groundTruth->motion[i] - guessB->motion[i]);
    if ( (diffA==0.0) && (diffA==diffB) )  { fprintf(stderr,BLUE  "%0.2f ",diffA-diffB); } else
    {
     if (diffA>=diffB)                     { fprintf(stderr,GREEN "%0.2f ",diffA-diffB); } else
                                           { fprintf(stderr,RED   "%0.2f ",diffB-diffA); }

     unsigned int jID =mc->motionToJointLookup[i].jointID;
     unsigned int chID=mc->motionToJointLookup[i].channelID;
     fprintf(stderr,NORMAL "(%s#%u/%0.2f->%0.2f/%0.2f) ",mc->jointHierarchy[jID].jointName,chID,guessA->motion[i],guessB->motion[i],groundTruth->motion[i]);
    }
  }
  fprintf(stderr,NORMAL "\n___________\n");
}












//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
///                                        PRINT STATE
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------
void bvh_printBVH(struct BVH_MotionCapture * bvhMotion)
{
  if (bvhMotion==0) {  fprintf(stdout,"\n\n\nCan't print empty BVH dataset..\n"); return; }
  fprintf(stdout,"\n\n\nPrinting BVH dataset..\n");

  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
    fprintf(stdout,"___________________________________\n");
    fprintf(stdout,GREEN "Joint %u - %s/%s " NORMAL ,i,bvhMotion->jointHierarchy[i].jointName,bvhMotion->jointHierarchy[i].jointNameLowercase);
    unsigned int parentID = bvhMotion->jointHierarchy[i].parentJoint;
    fprintf(stdout," | Parent %u - %s \n",parentID,bvhMotion->jointHierarchy[parentID].jointName);
    //===============================================================
    if (bvhMotion->jointHierarchy[i].loadedChannels>0)
    {
     fprintf(stdout,"Has %u channels - ",bvhMotion->jointHierarchy[i].loadedChannels);
     if ( bvhMotion->jointHierarchy[i].channelRotationOrder==0 ) { fprintf(stdout,RED "!");}
     fprintf(stdout,"Rotation Order: %s \n" NORMAL,rotationOrderNames[(unsigned int) bvhMotion->jointHierarchy[i].channelRotationOrder]);
     for (unsigned int z=0; z<bvhMotion->jointHierarchy[i].loadedChannels; z++)
      {
        unsigned int cT = bvhMotion->jointHierarchy[i].channelType[z];
        fprintf(stdout,"%s ",channelNames[cT]);
      }
     fprintf(stdout,"\n");
    } else
    {
     fprintf(stdout,"Has no channels\n");
    }
    //===============================================================
     fprintf(stdout,"Offset : ");
     for (unsigned int z=0; z<3; z++)
      {
        fprintf(stdout,"%0.5f ",bvhMotion->jointHierarchy[i].offset[z]);
      }
     fprintf(stdout,"\n");
    //===============================================================
    fprintf(stdout,"isRoot %u - ",(unsigned int) bvhMotion->jointHierarchy[i].isRoot);
    fprintf(stdout,"isEndSite %u - ",(unsigned int) bvhMotion->jointHierarchy[i].isEndSite);
    fprintf(stdout,"hasEndSite %u\n",(unsigned int) bvhMotion->jointHierarchy[i].hasEndSite);
    fprintf(stdout,"level %u\n",bvhMotion->jointHierarchy[i].hierarchyLevel );
    fprintf(stdout,"----------------------------------\n");
  }


  fprintf(stdout,"Motion data\n");
  fprintf(stdout,"___________________________________\n");
  fprintf(stdout,"Number of values per frame : %u \n",bvhMotion->numberOfValuesPerFrame);
  fprintf(stdout,"Loaded motion frames : %u \n",bvhMotion->numberOfFramesEncountered);
  fprintf(stdout,"Frame time : %0.8f \n",bvhMotion->frameTime);
  fprintf(stdout,"___________________________________\n");
}


void bvh_printBVHJointToMotionLookupTable(struct BVH_MotionCapture * bvhMotion)
{
  fprintf(stdout,"\n\n\nPrinting BVH JointToMotion lookup table..\n");
  fprintf(stdout,"_______________________________________________\n");
  for (unsigned int fID=0; fID<bvhMotion->numberOfFrames; fID++)
  {
   for (unsigned int jID=0; jID<bvhMotion->jointHierarchySize; jID++)
    {
     for (unsigned int channelNumber=0; channelNumber<bvhMotion->jointHierarchy[jID].loadedChannels; channelNumber++ )
     {
         unsigned int channelTypeID = bvhMotion->jointHierarchy[jID].channelType[channelNumber];
         unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,fID,channelTypeID);

         fprintf(stdout,"f[%u].%s.%s(%u)=%0.2f " ,
                 fID,
                 bvhMotion->jointHierarchy[jID].jointName,
                 channelNames[channelTypeID],
                 mID,
                 bvh_getMotionValue(bvhMotion,mID)
                 );
     }
    }
   fprintf(stdout,"\n\n");
  }
  fprintf(stdout,"_______________________________________________\n");
}



void bvh_print_C_Header(struct BVH_MotionCapture * bvhMotion)
{

  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief An array with BVH string labels\n");
  fprintf(stdout," */\n");
  fprintf(stdout,"static const char * BVHOutputArrayNames[] =\n");
  fprintf(stdout,"{\n");
  char comma=',';
  char coord;//='X'; This is overwritten so dont need to be assigned..
  unsigned int countOfChannels=0;
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
    if (i==0)
        {
           coord='X'; fprintf(stdout,"\"%s_%cposition\"%c // 0\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           coord='Y'; fprintf(stdout,"\"%s_%cposition\"%c // 1\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           coord='Z'; fprintf(stdout,"\"%s_%cposition\"%c // 2\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           coord='Z'; fprintf(stdout,"\"%s_%crotation\"%c // 3\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           coord='Y'; fprintf(stdout,"\"%s_%crotation\"%c // 4\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           coord='X'; fprintf(stdout,"\"%s_%crotation\"%c // 5\n",bvhMotion->jointHierarchy[i].jointName,coord,comma);
           countOfChannels+=5;
        } else
    {
     if (!bvhMotion->jointHierarchy[i].isEndSite)
        {
            for (unsigned int z=0; z<bvhMotion->jointHierarchy[i].loadedChannels; z++)
                {
                  ++countOfChannels;
                  if (countOfChannels+1>=bvhMotion->numberOfValuesPerFrame)
                  {
                      comma=' ';
                  }

                  unsigned int cT = bvhMotion->jointHierarchy[i].channelType[z];
                  fprintf(stdout,"\"%s_%s\"%c // %u\n ",bvhMotion->jointHierarchy[i].jointName,channelNames[cT],comma,countOfChannels);
                }

        }
    }
  }
  fprintf(stdout,"};\n\n\n\n");


  char label[513]={0};
  comma=',';
  //coord='X'; It is always reassigned before use..
  countOfChannels=0;

  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief This is a programmer friendly enumerator of joint output extracted from the BVH file.\n");
  fprintf(stdout," */\n");
  fprintf(stdout,"enum BVH_Output_Joints\n");
  fprintf(stdout,"{\n");
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
    if (i==0)
        {
           coord='X'; snprintf(label,512,"%s_%cposition",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s = 0,\n",label);

           coord='Y'; snprintf(label,512,"%s_%cposition",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//1 \n",label);

           coord='Z'; snprintf(label,512,"%s_%cposition",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//2 \n",label);

           coord='Z'; snprintf(label,512,"%s_%crotation",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//3 \n",label);

           coord='Y'; snprintf(label,512,"%s_%crotation",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//4 \n",label);

           coord='X'; snprintf(label,512,"%s_%crotation",bvhMotion->jointHierarchy[i].jointName,coord);
           uppercase(label);
           fprintf(stdout,"BVH_MOTION_%s,//5 \n",label);

           countOfChannels+=5;
        } else
        {
         if (!bvhMotion->jointHierarchy[i].isEndSite)
          {
            for (unsigned int z=0; z<bvhMotion->jointHierarchy[i].loadedChannels; z++)
                {
                  ++countOfChannels;
                  if (countOfChannels+1>=bvhMotion->numberOfValuesPerFrame)
                  {
                      comma=' ';
                  }

                  unsigned int cT = bvhMotion->jointHierarchy[i].channelType[z];
                  snprintf(label,512,"%s_%s",bvhMotion->jointHierarchy[i].jointName,channelNames[cT]);
                  uppercase(label);
                  fprintf(stdout,"BVH_MOTION_%s%c//%u \n",label,comma,countOfChannels);
                }
          }
        }
  }
  fprintf(stdout,"};\n\n\n");



  comma=',';
  //coord='X'; It is always reassigned before use..
  countOfChannels=0;

  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief This is a programmer friendly enumerator to access 3D output  extracted from the BVH file.\n");
  fprintf(stdout," */\n");
  fprintf(stdout,"enum BVH_3D_Output_Joints\n");
  fprintf(stdout,"{\n");
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
     snprintf(label,512,"%s",bvhMotion->jointHierarchy[i].jointName);
     uppercase(label);
     coord='X';
     fprintf(stdout,"BVH_3DPOINT_%s%c%c//%u \n",label,coord,comma,countOfChannels);
     ++countOfChannels;
     coord='Y';
     fprintf(stdout,"BVH_3DPOINT_%s%c%c//%u \n",label,coord,comma,countOfChannels);
     ++countOfChannels;
     coord='Z';
     fprintf(stdout,"BVH_3DPOINT_%s%c%c//%u \n",label,coord,comma,countOfChannels);
     ++countOfChannels;
  }
  fprintf(stdout,"};\n\n\n");





  fprintf(stdout,"/**\n");
  fprintf(stdout," * @brief An array with BVH string labels\n");
  fprintf(stdout," */\n");
  fprintf(stdout,"static const char * BVH3DPositionalOutputArrayNames[] =\n");
  fprintf(stdout,"{\n");
  comma=',';
  countOfChannels=0;
  for (unsigned int i=0; i<bvhMotion->jointHierarchySize; i++)
  {
           coord='X'; fprintf(stdout,"\"%s_%cposition\"%c // %u\n",bvhMotion->jointHierarchy[i].jointName,coord,comma,countOfChannels);
           ++countOfChannels;
           coord='Y'; fprintf(stdout,"\"%s_%cposition\"%c // %u\n",bvhMotion->jointHierarchy[i].jointName,coord,comma,countOfChannels);
           ++countOfChannels;
           coord='Z'; fprintf(stdout,"\"%s_%cposition\"%c // %u\n",bvhMotion->jointHierarchy[i].jointName,coord,comma,countOfChannels);
           ++countOfChannels;
  }
  fprintf(stdout,"};\n\n\n\n");


}
