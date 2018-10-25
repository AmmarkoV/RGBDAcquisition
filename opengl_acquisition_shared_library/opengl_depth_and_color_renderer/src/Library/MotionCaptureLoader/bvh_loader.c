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
#include "bvh_loader.h"
#include "../TrajectoryParser/InputParser_C.h"


const char * channelNames[] =
{
    "no rotation channel",
    "Xrotation",
    "Yrotation",
    "Zrotation",
    "Xposition",
    "Yposition",
    "Zposition",
//=================
    "End of Channel Names" ,
    "Unknown"
};

const char * rotationOrderNames[] =
{
  "no rotation order",
  "XYZ",
  "XZY",
  "YXZ",
  "YZX",
  "ZXY",
  "ZYX",
//=================
    "End of Channel Rotation Orders" ,
    "Unknown"
};


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

int enumerateInputParserChannel(struct InputParserC * ipc , unsigned int argumentNumber)
{
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Xrotation") ) {return BVH_ROTATION_X; } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Yrotation") ) {return BVH_ROTATION_Y; } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Zrotation") ) {return BVH_ROTATION_Z; } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Xposition") ) {return BVH_POSITION_X; } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Yposition") ) {return BVH_POSITION_Y; } else
  if ( InputParser_WordCompareNoCaseAuto(ipc,argumentNumber,"Zposition") ) {return BVH_POSITION_Z; }

  return BVH_POSITION_NONE;
}


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
    fprintf(stderr,"BUG: Channel order still wrong, todo smarter channel order enumeration..\n");
  }
return channelOrder;
}

int getParentJoint(struct BVH_MotionCapture * bvhMotion , unsigned int currentJoint , unsigned int hierarchyLevel , unsigned int * parentJoint)
{
  if (currentJoint>=bvhMotion->MAX_jointHierarchySize)
  {
    fprintf(stderr,"getParentJoint: Incorrect currentJoint value\n");
    return 0;
  }

  if (hierarchyLevel==0)
  {
   //Already at root level..
   //No Parent joint..
   *parentJoint = 0;
   return 1;
  }

  unsigned int i = currentJoint;

  while (i>0)
  {
    if (bvhMotion->jointHierarchy[i].hierarchyLevel==hierarchyLevel-1)
    {
      //Found Parent Joint..!
      *parentJoint = i;
      return 1;
    }
    --i;
  }


  //We did not find something better than the root joint..
  *parentJoint = 0;
  return 1;
  //return 0;
}

int readBVHHeader(struct BVH_MotionCapture * bvhMotion , FILE * fd )
{
  bvhMotion->numberOfValuesPerFrame = 0;//57;
  bvhMotion->MAX_jointHierarchySize = MAX_BVH_JOINT_HIERARCHY_SIZE;

  int done=0;
  int atHeaderSection=0;
  ssize_t read;

  if (fd!=0)
  {
   struct InputParserC * ipc = InputParser_Create(1024,4);
   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,'[');
   InputParser_SetDelimeter(ipc,2,']');
   InputParser_SetDelimeter(ipc,3,'\n');

   struct InputParserC * ipcB = InputParser_Create(1024,3);
   InputParser_SetDelimeter(ipcB,0,' ');
   InputParser_SetDelimeter(ipcB,1,'\t');
   InputParser_SetDelimeter(ipcB,2,'\n');

    unsigned int i=0;
    unsigned int jNum=0; //this is used internally instead of jointHierarchySize to make code more readable
    unsigned int lookupID=0;
    unsigned int currentJoint=0; //this is used internally instead of jointHierarchySize to make code more readable
    unsigned int hierarchyLevel=0;
    char * line = NULL;
    size_t len = 0;

    while  ( (!done) && ((read = getline(&line, &len, fd)) != -1) )
    {
       //printf("Retrieved line of length %zu :\n", read);
       //printf("%s", line);
       int num = InputParser_SeperateWords(ipc,line,1);


      if (num>0)
      { //We have content..
       if (!atHeaderSection)
       {
          //We won't start parsing unless we reach a HIERARCHY line
          if (InputParser_WordCompareAuto(ipc,0,"HIERARCHY"))  { atHeaderSection=1; }
       } else
       {
         int numB = InputParser_SeperateWords(ipcB,line,1);
         if (numB>0)
         {
         if (InputParser_WordCompareAuto(ipcB,0,"ROOT"))
              {
               //We encountered something like |ROOT Hips|
               fprintf(stderr,"-R-");
               //Store new ROOT Joint Name
               InputParser_GetWord(
                                    ipcB,1,
                                    bvhMotion->jointHierarchy[jNum].jointName ,
                                    MAX_BVH_JOINT_NAME
                                  );
               fprintf(stderr,"-%s-",bvhMotion->jointHierarchy[jNum].jointName);
               //Store new Joint Hierarchy Level
               //Rest of the information will be filled in when we reach an {
               bvhMotion->jointHierarchy[jNum].hierarchyLevel = hierarchyLevel;
               bvhMotion->jointHierarchy[jNum].isRoot=1;
               bvhMotion->jointHierarchy[jNum].lookupID = bvhMotion->numberOfValuesPerFrame;
               currentJoint=jNum;
               ++jNum;
             } else
         if (InputParser_WordCompareAuto(ipcB,0,"JOINT"))
              {
               //We encountered something like |JOINT Chest|
               fprintf(stderr,"-J-");
               //Store new Joint Name
               InputParser_GetWord(
                                    ipcB,1,
                                    bvhMotion->jointHierarchy[jNum].jointName ,
                                    MAX_BVH_JOINT_NAME
                                  );
               fprintf(stderr,"-%s-",bvhMotion->jointHierarchy[jNum].jointName);
               //Store new Joint Hierarchy Level
               //Rest of the information will be filled in when we reach an {
               bvhMotion->jointHierarchy[jNum].hierarchyLevel = hierarchyLevel;
               bvhMotion->jointHierarchy[jNum].lookupID = bvhMotion->numberOfValuesPerFrame;
               currentJoint=jNum;
               ++jNum;
             } else
         if (InputParser_WordCompareAuto(ipcB,0,"End"))
             {
               //We encountered something like |End Site|
              fprintf(stderr,"-E-");
              if (InputParser_WordCompareAuto(ipcB,1,"Site"))
                   {
                    fprintf(stderr,"-S-");
                    snprintf(bvhMotion->jointHierarchy[jNum].jointName,MAX_BVH_JOINT_NAME,"End Site");
                    bvhMotion->jointHierarchy[jNum].isEndSite=1;
                    bvhMotion->jointHierarchy[jNum].hierarchyLevel = hierarchyLevel;
                    bvhMotion->jointHierarchy[jNum].lookupID = bvhMotion->numberOfValuesPerFrame;
                    currentJoint=jNum;
                    ++jNum;
                   }
              } else
         if (InputParser_WordCompareAuto(ipcB,0,"CHANNELS"))
             { //Reached something like  |CHANNELS 3 Zrotation Xrotation Yrotation| declaration
              fprintf(stderr,"-C-");

              //Read number of Channels
              bvhMotion->jointHierarchy[currentJoint].loadedChannels=InputParser_GetWordInt(ipcB,1);
              fprintf(stderr,"-%u-",bvhMotion->jointHierarchy[currentJoint].loadedChannels);
              //Sum the number of channels for motion commands later..
              bvhMotion->numberOfValuesPerFrame += bvhMotion->jointHierarchy[currentJoint].loadedChannels;

              //First to wipe channels to make sure they are clean
              memset(bvhMotion->jointHierarchy[currentJoint].channels,0,sizeof(float) * BVH_VALID_CHANNEL_NAMES);
              memset(bvhMotion->jointHierarchy[currentJoint].channelType,0,sizeof(char) * BVH_VALID_CHANNEL_NAMES);
              //Now to store the channel labels
              for (i=0; i<bvhMotion->jointHierarchy[currentJoint].loadedChannels; i++)
                  {
                   //For each declared channel we need to enumerate the label to a value
                   unsigned int channelID = enumerateInputParserChannel(ipcB,2+i);
                   bvhMotion->jointHierarchy[currentJoint].channelType[i]=channelID;
                   fprintf(stderr,"-%u-",bvhMotion->jointHierarchy[currentJoint].channelType[i]);
                   //Store at channel lookup table
                   unsigned int parentID=bvhMotion->jointHierarchy[currentJoint].parentJoint;
                   bvhMotion->lookupTable[lookupID].channelID = channelID;
                   bvhMotion->lookupTable[lookupID].jointID   = currentJoint;
                   bvhMotion->lookupTable[lookupID].parentID  = parentID;
                  }

               bvhMotion->jointHierarchy[currentJoint].channelRotationOrder =enumerateChannelOrder(bvhMotion,currentJoint);
              //Done
              } else
         if (InputParser_WordCompareAuto(ipcB,0,"OFFSET"))
             {//Reached something like |OFFSET	 3.91	 0.00	 0.00|
              fprintf(stderr,"-O-");

              //Store offsets..
              //TODO: could check numB to make sure all offsets are present..
              bvhMotion->jointHierarchy[currentJoint].offset[0]=InputParser_GetWordFloat(ipcB,1);
              bvhMotion->jointHierarchy[currentJoint].offset[1]=InputParser_GetWordFloat(ipcB,2);
              bvhMotion->jointHierarchy[currentJoint].offset[2]=InputParser_GetWordFloat(ipcB,3);
             } else
         if (InputParser_WordCompareAuto(ipcB,0,"{"))
             {
              //We reached an { so we need to finish our joint OR root declaration
              fprintf(stderr,"-{%u-",hierarchyLevel);
              if (
                  bvhMotion->jointHierarchy[currentJoint].hierarchyLevel == hierarchyLevel
                 )
                 {
                  bvhMotion->jointHierarchy[currentJoint].hasBrace=1;

                  if (
                      getParentJoint(
                                     bvhMotion,
                                     currentJoint,
                                     hierarchyLevel,
                                     &bvhMotion->jointHierarchy[currentJoint].parentJoint
                                    )
                      )
                   {
                     //We have a parent joint..!
                     if ( bvhMotion->jointHierarchy[currentJoint].isEndSite)
                      { //If current joint is an EndSite we must inform parent joint that it has an End Site
                       unsigned int parentID=bvhMotion->jointHierarchy[currentJoint].parentJoint;
                       bvhMotion->jointHierarchy[parentID].hasEndSite=1;
                      }
                   } else
                   {
                    fprintf(stderr,"Bug: could not find a parent joint..\n");
                   }
                 } else
                 {
                  fprintf(stderr,"Bug: HierarchyLevel not set at braces..\n");
                 }
               ++hierarchyLevel;
              } else
         if (InputParser_WordCompareAuto(ipcB,0,"}"))
             {
              //We reached an } so we pop one hierarchyLevel
              if (hierarchyLevel>0)
                 {
                  --hierarchyLevel;
                 } else
                 {
                   fprintf(stderr,"Erroneous BVH hierarchy..\n");
                 }

              if (hierarchyLevel==0)
              {
                //We are done..
                done=1;
              }

              //-------------------------------------
              fprintf(stderr,"-%u}-",hierarchyLevel);
             }
          else
         {
            //Unexpected input..
            printf("Unexpected line of length %zu :\n", read);
            printf("%s", line);
         }

         } // We have header content
       } // We are at header section
      } //We have content
    } //We have line input from file

   if (hierarchyLevel!=0)
   {
     fprintf(stderr,"Missing } braces..\n");
     atHeaderSection = 0;
   }

   bvhMotion->jointHierarchySize = jNum;

   InputParser_Destroy(ipc);
   InputParser_Destroy(ipcB);
  }


  fprintf(
           stderr,
           "\nNumber of Values Per Frame: %u\n",
           bvhMotion->numberOfValuesPerFrame
          );

 return atHeaderSection;
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

int pushNewBVHMotionState(struct BVH_MotionCapture * bvhMotion , char * parameters)
{
   if (
         (bvhMotion->motionValues==0) ||
         (bvhMotion->motionValuesSize==0)
      )
   {
     fprintf(stderr,"cannot pushNewBVHMotionState without space to store new information\n");
     return 0;
   }

   struct InputParserC * ipc = InputParser_Create(1024,5);
   if (ipc==0) { return 0; }

   InputParser_SetDelimeter(ipc,0,' ');
   InputParser_SetDelimeter(ipc,1,'\t');
   InputParser_SetDelimeter(ipc,2,' ');
   InputParser_SetDelimeter(ipc,3,' ');
   InputParser_SetDelimeter(ipc,3,'\n');

   unsigned int i=0;
   int numberOfParameters = InputParser_SeperateWords(ipc,parameters,1);
   fprintf(stderr,"MOTION command has %u parameters\n",numberOfParameters);

   if (numberOfParameters>0)
   {
     if (numberOfParameters + bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame < bvhMotion->motionValuesSize+1)
     {
      fprintf(stderr,
              "Filling from %u to %u \n",
              bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame,
              numberOfParameters+bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame
             );

      for (i=0; i<numberOfParameters; i++)
      {
        //fprintf(stderr,"P%u=%0.2f ",i,InputParser_GetWordFloat(ipc,i));
        bvhMotion->motionValues[i+bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame] = InputParser_GetWordFloat(ipc,i);
      }
     }
     bvhMotion->numberOfFramesEncountered++;
   }

   InputParser_Destroy(ipc);
   return 1;
}






int readBVHMotion(struct BVH_MotionCapture * bvhMotion , FILE * fd )
{
  int atMotionSection=0;
  ssize_t read;

  if (fd!=0)
  {
   struct InputParserC * ipc = InputParser_Create(1024,5);

   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,'[');
   InputParser_SetDelimeter(ipc,2,',');
   InputParser_SetDelimeter(ipc,3,']');
   InputParser_SetDelimeter(ipc,4,'\n');

    char str[512];
    char * line = NULL;
    size_t len = 0;

    while ((read = getline(&line, &len, fd)) != -1)
    {
       printf("Retrieved line of length %zu :\n", read);
       printf("%s", line);
       int num = InputParser_SeperateWords(ipc,line,1);

       //InputParser_GetWord(ipc,0,str,512);
       //fprintf(stderr,"Word0=`%s`",str);
       //InputParser_GetWord(ipc,1,str,512);
       //fprintf(stderr,"Word1=`%s`",str);
       //InputParser_GetWord(ipc,2,str,512);
       //fprintf(stderr,"Word2=`%s`",str);
      if (num>0)
      { //We have content..
       if (!atMotionSection)
       {
          if (InputParser_WordCompareAuto(ipc,0,"MOTION"))      { atMotionSection=1; }
       } else
       {
         if (InputParser_WordCompareAuto(ipc,0,"Frames"))       { bvhMotion->numberOfFrames = InputParser_GetWordInt(ipc,1); } else
         if (InputParser_WordCompareAuto(ipc,0,"Frame Time"))   { bvhMotion->frameTime = InputParser_GetWordFloat(ipc,1); }      else
         {
           if (bvhMotion->motionValues==0)
           {
             //If we haven't yet allocated a motionValues array we need to do so now..!
             bvhMotion->motionValuesSize = bvhMotion->numberOfFrames * bvhMotion->numberOfValuesPerFrame;
             bvhMotion->motionValues = (float*)  malloc(sizeof(float) * (1+bvhMotion->motionValuesSize));
           }

           //This is motion input
           InputParser_GetWord(ipc,0,str,512);
           pushNewBVHMotionState(bvhMotion,str);
           str[0]=0;//Clean up str
         }
       }
       }
    }

   InputParser_Destroy(ipc);
  }

  fprintf(
           stderr,
           "Frames: %u(%u) / Frame Time : %0.4f\n",
           bvhMotion->numberOfFrames,
           bvhMotion->numberOfFramesEncountered,
           bvhMotion->frameTime
          );

  return (atMotionSection);
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
void bvh_printBVH(struct BVH_MotionCapture * bvhMotion)
{
  fprintf(stderr,"\n\n\nPrinting BVH file..\n");
  int i=0,z=0;
  for (i=0; i<bvhMotion->jointHierarchySize; i++)
  {
    fprintf(stderr,"___________________________________\n");
    fprintf(stderr,"Joint %u - %s \n",i,bvhMotion->jointHierarchy[i].jointName);
    fprintf(stderr,"___________________________________\n");
    unsigned int parentID = bvhMotion->jointHierarchy[i].parentJoint;
    fprintf(stderr,"Parent %u - %s \n",parentID,bvhMotion->jointHierarchy[parentID].jointName);
    //===============================================================
    if (bvhMotion->jointHierarchy[i].loadedChannels>0)
    {
     fprintf(stderr,"Has %u channels\n",bvhMotion->jointHierarchy[i].loadedChannels);
     fprintf(stderr,"Rotation Order: %s \n",rotationOrderNames[(unsigned int) bvhMotion->jointHierarchy[i].channelRotationOrder]);
     for (z=0; z<bvhMotion->jointHierarchy[i].loadedChannels; z++)
      {
        unsigned int cT = bvhMotion->jointHierarchy[i].channelType[z];
        fprintf(stderr,"%s ",channelNames[cT]);
      }
     fprintf(stderr,"\n");
    } else
    {
     fprintf(stderr,"Has no channels\n");
    }
    //===============================================================
     fprintf(stderr,"Offset : ");
     for (z=0; z<3; z++)
      {
        fprintf(stderr,"%0.5f ",bvhMotion->jointHierarchy[i].offset[z]);
      }
     fprintf(stderr,"\n");
    //===============================================================
    fprintf(stderr,"isRoot %u \n",bvhMotion->jointHierarchy[i].isRoot);
    fprintf(stderr,"isEndSite %u \n",bvhMotion->jointHierarchy[i].isEndSite);
    fprintf(stderr,"hasEndSite %u\n",bvhMotion->jointHierarchy[i].hasEndSite);
    fprintf(stderr,"----------------------------------\n");
  }


  fprintf(stderr,"Motion data\n");
  fprintf(stderr,"___________________________________\n");
  fprintf(stderr,"loaded motion frames : %u \n",bvhMotion->numberOfFramesEncountered);
  fprintf(stderr,"frame time : %0.8f \n",bvhMotion->frameTime);
  fprintf(stderr,"___________________________________\n");
}


int bvh_loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion)
{
  int successfullRead=0;
  FILE *fd=0;
  fd = fopen(filename,"r");
  if (fd!=0)
    {
      snprintf(bvhMotion->fileName,1024,"%s",filename);
      if (readBVHHeader(bvhMotion,fd))
      {
       if (readBVHMotion(bvhMotion,fd))
       {
         successfullRead=1;
       }
      }
      fclose(fd);
    }
 return successfullRead;
}

int bvh_getJointIDFromJointName(
                                 struct BVH_MotionCapture * bvhMotion ,
                                 const char * jointName,
                                 BVHJointID * jID
                                )
{
   if (bvhMotion==0) { return 0; }
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


float * bvh_getJointOffset(struct BVH_MotionCapture * bvhMotion , BVHJointID jID)
{
   if (bvhMotion==0) { return 0; }
   if (bvhMotion->jointHierarchySize<=jID) { return 0; }

   return bvhMotion->jointHierarchy[jID].offset;
}


float  bvh_getJointChannelAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID)
{
   if (bvhMotion==0) { return 0.0; }
   if (bvhMotion->jointHierarchySize<=jID) { return 0.0; }

   unsigned int mID = fID * bvhMotion->numberOfValuesPerFrame + bvhMotion->jointHierarchy[jID].lookupID;


   /*
   fprintf(stderr,"Joint %s - ", bvhMotion->jointHierarchy[jID].jointName);
   fprintf(stderr,"LookupID %u ", bvhMotion->jointHierarchy[jID].lookupID);
   fprintf(stderr,"FrameID %u ", fID);
   fprintf(stderr,"MotionID %u \n", mID);
   */


   unsigned int i=0;
   for (i=0; i<BVH_VALID_CHANNEL_NAMES; i++)
   {
       if ( bvhMotion->jointHierarchy[jID].channelType[i] == channelTypeID)
       {
         mID+=i;
         break;
       }
   }

   if (mID>=bvhMotion->motionValuesSize)
   {
     return 0.0;
   }

   return bvhMotion->motionValues[mID];
}



float  bvh_getJointRotationXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID)
{
  return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_X);
}


float  bvh_getJointRotationYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID)
{
  return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Y);
}


float  bvh_getJointRotationZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID)
{
  return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_ROTATION_Z);
}



float  bvh_getJointPositionXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID)
{
  return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_X);
}


float  bvh_getJointPositionYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID)
{
  return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Y);
}


float  bvh_getJointPositionZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID)
{
  return bvh_getJointChannelAtFrame(bvhMotion,jID,fID,BVH_POSITION_Z);
}


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


int bhv_jointHasParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID )
{
    if (jID>bvhMotion->jointHierarchySize) { return 0; }
    return (!bvhMotion->jointHierarchy[jID].isRoot);
}
