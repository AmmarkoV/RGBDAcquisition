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


void lowercase(char *a)
{
    for (;; a++)
        {
         *a = tolower((unsigned char)*a);
        }
}

void bvh_renameJoints(struct BVH_MotionCapture * bvhMotion)
{
  unsigned int jID=0;

  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
     char * jN = bvhMotion->jointHierarchy[jID].jointName;

     lowercase(jN);

     //-------------------------------------------------------------------------------------------------
     if  ( (strcmp(jN,"hip")==0) || (strcmp(jN,"hips")==0) )
           { snprintf(jN,MAX_BVH_JOINT_NAME,"hip"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"abdomen")==0) || (strcmp(jN,"spine")==0) || (strcmp(jN,"torso")==0)  || (strcmp(jN,"waist")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"abdomen"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"chest")==0) || (strcmp(jN,"chest2")==0) || (strcmp(jN,"spine1")==0)  || (strcmp(jN,"spine2")==0) || (strcmp(jN,"torso2")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"chest"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"lefthip")==0) || (strcmp(jN,"leftupleg")==0) || (strcmp(jN,"lthigh")==0)  || (strcmp(jN,"leftupperLeg")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lthigh"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftknee")==0) || (strcmp(jN,"leftlowleg")==0) || (strcmp(jN,"leftleg")==0)  || (strcmp(jN,"lshin")==0) || (strcmp(jN,"leftlowerleg")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lknee"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftankle")==0) || (strcmp(jN,"leftfoot")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lfoot"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"righthip")==0) || (strcmp(jN,"rightupleg")==0) || (strcmp(jN,"rthigh")==0)  || (strcmp(jN,"rightupperleg")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rthigh"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightknee")==0) || (strcmp(jN,"rightlowleg")==0) || (strcmp(jN,"rightleg")==0)  || (strcmp(jN,"rshin")==0) || (strcmp(jN,"rightlowerleg")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rknee"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightankle")==0) || (strcmp(jN,"rightfoot")==0) || (strcmp(jN,"rfoot")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rfoot"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftcollar")==0) || (strcmp(jN,"lcollar")==0) || (strcmp(jN,"leftshoulder")==0)  || (strcmp(jN,"lcolr")==0) || (strcmp(jN,"rightlowerleg")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lcollar"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftuparm")==0) || (strcmp(jN,"leftarm")==0) || (strcmp(jN,"lshldr")==0)  || (strcmp(jN,"leftshoulder")==0) || (strcmp(jN,"leftupperarm")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lshoulder"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftelbow")==0) || (strcmp(jN,"leftlowarm")==0) || (strcmp(jN,"leftforearm")==0)  || (strcmp(jN,"lforearm")==0) || (strcmp(jN,"leftlowerarm")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lelbow"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftwrist")==0) || (strcmp(jN,"lefthand")==0) || (strcmp(jN,"lhand")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lhand"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightcollar")==0) || (strcmp(jN,"rcollar")==0) || (strcmp(jN,"rightshoulder")==0)  || (strcmp(jN,"rcolr")==0) || (strcmp(jN,"rightlowerleg")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rcollar"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightuparm")==0) || (strcmp(jN,"rightarm")==0) || (strcmp(jN,"rshldr")==0)  || (strcmp(jN,"rightshoulder")==0) || (strcmp(jN,"rightupperarm")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rshoulder"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightelbow")==0) || (strcmp(jN,"rightlowarm")==0) || (strcmp(jN,"rightforearm")==0)  || (strcmp(jN,"rforearm")==0) || (strcmp(jN,"rightlowerarm")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"relbow"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightwrist")==0) || (strcmp(jN,"righthand")==0) || (strcmp(jN,"rhand")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rhand"); }
     //-------------------------------------------------------------------------------------------------
   }
}


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
    fprintf(stderr,RED "BUG: Channel order still wrong, todo smarter channel order enumeration..\n" NORMAL);
  }
return channelOrder;
}


unsigned int bvh_resolveFrameAndJointAndChannelToMotionID(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID)
{
   if (channelTypeID>=BVH_VALID_CHANNEL_NAMES)
   {
     fprintf(stderr,RED "bvh_resolveFrameAndJointAndChannelToMotionID: asked to resolve non-existing channel type\n" NORMAL);
     return 0;
   }

   if (jID>=bvhMotion->jointHierarchySize)
   {
     fprintf(stderr,RED "bvh_resolveFrameAndJointAndChannelToMotionID: asked to resolve non-existing joint\n" NORMAL);
     return 0;
   }

   return  (fID * bvhMotion->numberOfValuesPerFrame) + bvhMotion->jointToMotionLookup[jID].channelIDMotionOffset[channelTypeID];
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

int thisLineOnlyHasX(const char * line,const char x)
{
  unsigned int i=0;
  for (i=0; i<strlen(line); i++)
  {
    if (line[i]==' ')  { } else
    if (line[i]=='\t') { } else
    if (line[i]==10) { } else
    if (line[i]==13) { } else
    if (line[i]==x)  { } else
        {
           //fprintf(stderr,"Line Char %u is %u(%c)\n",i,(unsigned int) line[i],line[i]);
           return 0;
        }
  }
 return 1;
}


int readBVHHeader(struct BVH_MotionCapture * bvhMotion , FILE * fd )
{
  bvhMotion->linesParsed=0;
  bvhMotion->numberOfValuesPerFrame = 0;//57;
  bvhMotion->MAX_jointHierarchySize = MAX_BVH_JOINT_HIERARCHY_SIZE;

  int done=0;
  int atHeaderSection=0;
  ssize_t read;

  if (fd!=0)
  {
   struct InputParserC * ipc = InputParser_Create(4096,5);
   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,'[');
   InputParser_SetDelimeter(ipc,2,']');
   InputParser_SetDelimeter(ipc,3,10);
   InputParser_SetDelimeter(ipc,4,13);

   struct InputParserC * ipcB = InputParser_Create(4096,4);
   InputParser_SetDelimeter(ipcB,0,' ');
   InputParser_SetDelimeter(ipcB,1,'\t');
   InputParser_SetDelimeter(ipcB,2,10);
   InputParser_SetDelimeter(ipcB,3,13);

    unsigned int jNum=0; //this is used internally instead of jointHierarchySize to make code more readable
    unsigned int currentJoint=0; //this is used internally instead of jointHierarchySize to make code more readable
    unsigned int hierarchyLevel=0;
    char * line = NULL;
    size_t len = 0;

    while  ( (!done) && ((read = getline(&line, &len, fd)) != -1) )
    {
       ++bvhMotion->linesParsed;
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
               //Update lookup table to remember ordering
               bvhMotion->jointToMotionLookup[jNum].jointMotionOffset  = bvhMotion->numberOfValuesPerFrame;

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
               //Update lookup table to remember ordering
               bvhMotion->jointToMotionLookup[jNum].jointMotionOffset  = bvhMotion->numberOfValuesPerFrame;
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
                    //Update lookup table to remember ordering
                    bvhMotion->jointToMotionLookup[jNum].jointMotionOffset  = bvhMotion->numberOfValuesPerFrame;
                    currentJoint=jNum;
                    ++jNum;
                   }
              } else
         if (InputParser_WordCompareAuto(ipcB,0,"CHANNELS"))
             { //Reached something like  |CHANNELS 3 Zrotation Xrotation Yrotation| declaration
              fprintf(stderr,"-C");

              //Keep as shorthand..
              unsigned int parentID=bvhMotion->jointHierarchy[currentJoint].parentJoint;

              //Read number of Channels
              unsigned int loadedChannels = InputParser_GetWordInt(ipcB,1);
              bvhMotion->jointHierarchy[currentJoint].loadedChannels=loadedChannels;
              fprintf(stderr,"(%u)-",loadedChannels);

              //First wipe channels to make sure they are clean
              memset(bvhMotion->jointHierarchy[currentJoint].channelType,0,sizeof(char) * BVH_VALID_CHANNEL_NAMES);

              fprintf(stderr,"\nJOINT %u (%s) : ",currentJoint,bvhMotion->jointHierarchy[currentJoint].jointName);

              //Now to store the channel labels
              unsigned int cL=0; //Channel To Load
              for (cL=0; cL<loadedChannels; cL++)
                  {
                   //For each declared channel we need to enumerate the label to a value
                   unsigned int thisChannelID = enumerateInputParserChannel(ipcB,2+cL);

                   bvhMotion->jointHierarchy[currentJoint].channelType[cL]=thisChannelID;

                   fprintf(stderr,"#%u %s=%u ",cL,channelNames[thisChannelID],bvhMotion->numberOfValuesPerFrame);

                   //Update jointToMotion Lookup Table..
                   bvhMotion->jointToMotionLookup[currentJoint].channelIDMotionOffset[thisChannelID] = bvhMotion->numberOfValuesPerFrame;

                   //Update motionToJoint Lookup Table..
                   bvhMotion->motionToJointLookup[bvhMotion->numberOfValuesPerFrame].channelID = thisChannelID;
                   bvhMotion->motionToJointLookup[bvhMotion->numberOfValuesPerFrame].jointID   = currentJoint;
                   bvhMotion->motionToJointLookup[bvhMotion->numberOfValuesPerFrame].parentID  = parentID;

                   ++bvhMotion->numberOfValuesPerFrame;
                  }
                fprintf(stderr,"\n");

               bvhMotion->jointHierarchy[currentJoint].channelRotationOrder = enumerateChannelOrder(bvhMotion,currentJoint);
              //Done
              } else
         if (InputParser_WordCompareAuto(ipcB,0,"OFFSET"))
             {//Reached something like |OFFSET	 3.91	 0.00	 0.00|
              fprintf(stderr,"-O-");

              //Store offsets..
              //TODO: could check numB to make sure all offsets are present..
              if (numB==4)
                {
                 bvhMotion->jointHierarchy[currentJoint].offset[0]=InputParser_GetWordFloat(ipcB,1);
                 bvhMotion->jointHierarchy[currentJoint].offset[1]=InputParser_GetWordFloat(ipcB,2);
                 bvhMotion->jointHierarchy[currentJoint].offset[2]=InputParser_GetWordFloat(ipcB,3);
                } else
                {
                 fprintf(stderr,RED "Incorrect number of offset arguments..\n" NORMAL);
                 bvhMotion->jointHierarchy[currentJoint].offset[0]=0.0;
                 bvhMotion->jointHierarchy[currentJoint].offset[1]=0.0;
                 bvhMotion->jointHierarchy[currentJoint].offset[2]=0.0;
                }

                 double * m = bvhMotion->jointHierarchy[currentJoint].staticTransformation;
                 m[0] =1.0;  m[1] =0.0;  m[2] =0.0;  m[3] = (double) bvhMotion->jointHierarchy[currentJoint].offset[0];
                 m[4] =0.0;  m[5] =1.0;  m[6] =0.0;  m[7] = (double) bvhMotion->jointHierarchy[currentJoint].offset[1];
                 m[8] =0.0;  m[9] =0.0;  m[10]=1.0;  m[11]= (double) bvhMotion->jointHierarchy[currentJoint].offset[2];
                 m[12]=0.0;  m[13]=0.0;  m[14]=0.0;  m[15]=1.0;


             } else
         if ( (InputParser_WordCompareAuto(ipcB,0,"{")) || (thisLineOnlyHasX(line,'{')) )
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
                    fprintf(stderr,RED "Bug: could not find a parent joint..\n" NORMAL);
                   }
                 } else
                 {
                  fprintf(stderr,RED "Bug: HierarchyLevel not set at braces..\n" NORMAL );
                 }
               ++hierarchyLevel;
              } else
         if ( (InputParser_WordCompareAuto(ipcB,0,"}")) || (thisLineOnlyHasX(line,'}') ) )
             {
              //We reached an } so we pop one hierarchyLevel
              if (hierarchyLevel>0)
                 {
                  --hierarchyLevel;
                 } else
                 {
                   fprintf(stderr,RED "Erroneous BVH hierarchy..\n" NORMAL);
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
            fprintf(stderr,"Unexpected line num (%u) of length %zu :\n" , bvhMotion->linesParsed , read);
            fprintf(stderr,"%s", line);
         }

         } // We have header content
       } // We are at header section
      } //We have content
    } //We have line input from file

   //Free incoming line buffer..
   if (line) { free(line); }

   if (hierarchyLevel!=0)
   {
     fprintf(stderr,RED "Missing } braces..\n" NORMAL);
     atHeaderSection = 0;
   }

   bvhMotion->jointHierarchySize = jNum;

   InputParser_Destroy(ipc);
   InputParser_Destroy(ipcB);
   //We need file to be open..!
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

int pushNewBVHMotionState(struct BVH_MotionCapture * bvhMotion ,const char * parameters)
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
   InputParser_SetDelimeter(ipc,3,10);
   InputParser_SetDelimeter(ipc,4,13);

   unsigned int i=0;
   unsigned int numberOfParameters = InputParser_SeperateWordsCC(ipc,parameters,1);
   //fprintf(stderr,"MOTION command has %u parameters\n",numberOfParameters);


   if (numberOfParameters==bvhMotion->numberOfValuesPerFrame)
   {
     if (numberOfParameters + bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame < bvhMotion->motionValuesSize+1)
     {
      /*
      fprintf(stderr,
              "Filling from %u to %u \n",
              bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame,
              numberOfParameters+bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame
             );*/

      for (i=0; i<numberOfParameters; i++)
      {
        //fprintf(stderr,"P%u=%0.2f ",i,InputParser_GetWordFloat(ipc,i));
        bvhMotion->motionValues[i+bvhMotion->numberOfFramesEncountered  * bvhMotion->numberOfValuesPerFrame] = InputParser_GetWordFloat(ipc,i);
      }
     }
     bvhMotion->numberOfFramesEncountered++;
   } else
   {
    //Unexpected input..
    fprintf(stderr,"Motion Expected had %u parameters we received %u\n",bvhMotion->numberOfValuesPerFrame,numberOfParameters);
    fprintf(stderr,"Unexpected line num (%u)  :\n" , bvhMotion->linesParsed);
    fprintf(stderr,"%s", parameters);
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
   struct InputParserC * ipc = InputParser_Create(2048,3);

   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,10);
   InputParser_SetDelimeter(ipc,2,13);

    char str[2048]={0};
    char * line = NULL;
    size_t len = 0;

    while ((read = getline(&line, &len, fd)) != -1)
    {
       ++bvhMotion->linesParsed;

       //fprintf(stderr,"Retrieved line of length %zu :\n", read);
       //fprintf(stderr,"%s", line);

       int num = InputParser_SeperateWords(ipc,line,1);

       //fprintf(stderr,"Has %u arguments\n",num);
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
          if (InputParser_WordCompareAuto(ipc,0,"MOTION"))      { fprintf(stderr,"Found Motion Section..\n"); atMotionSection=1; }
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
           InputParser_GetWord(ipc,0,str,2048);
           pushNewBVHMotionState(bvhMotion,str);
           str[0]=0;//Clean up str
         }
       }
       }
    }

   //Free incoming line buffer..
   if (line) { free(line); }

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
  unsigned int i=0,z=0;
  for (i=0; i<bvhMotion->jointHierarchySize; i++)
  {
    fprintf(stderr,"___________________________________\n");
    fprintf(stderr,GREEN "Joint %u - %s " NORMAL ,i,bvhMotion->jointHierarchy[i].jointName);
    unsigned int parentID = bvhMotion->jointHierarchy[i].parentJoint;
    fprintf(stderr," | Parent %u - %s \n",parentID,bvhMotion->jointHierarchy[parentID].jointName);
    //===============================================================
    if (bvhMotion->jointHierarchy[i].loadedChannels>0)
    {
     fprintf(stderr,"Has %u channels - ",bvhMotion->jointHierarchy[i].loadedChannels);
     if ( bvhMotion->jointHierarchy[i].channelRotationOrder==0 ) { fprintf(stderr,RED "!");}
     fprintf(stderr,"Rotation Order: %s \n" NORMAL,rotationOrderNames[(unsigned int) bvhMotion->jointHierarchy[i].channelRotationOrder]);
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
    fprintf(stderr,"isRoot %u - ",bvhMotion->jointHierarchy[i].isRoot);
    fprintf(stderr,"isEndSite %u - ",bvhMotion->jointHierarchy[i].isEndSite);
    fprintf(stderr,"hasEndSite %u\n",bvhMotion->jointHierarchy[i].hasEndSite);
    fprintf(stderr,"----------------------------------\n");
  }


  fprintf(stderr,"Motion data\n");
  fprintf(stderr,"___________________________________\n");
  fprintf(stderr,"Number of values per frame : %u \n",bvhMotion->numberOfValuesPerFrame);
  fprintf(stderr,"Loaded motion frames : %u \n",bvhMotion->numberOfFramesEncountered);
  fprintf(stderr,"Frame time : %0.8f \n",bvhMotion->frameTime);
  fprintf(stderr,"___________________________________\n");
}


void bvh_printBVHJointToMotionLookupTable(struct BVH_MotionCapture * bvhMotion)
{
  fprintf(stderr,"\n\n\nPrinting BVH JointToMotion lookup table..\n");
  fprintf(stderr,"_______________________________________________\n");
  unsigned int jID=0,fID=0,channelNumber;
  for (fID=0; fID<bvhMotion->numberOfFrames; fID++)
  {
   for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
    {
     for (channelNumber=0; channelNumber<bvhMotion->jointHierarchy[jID].loadedChannels; channelNumber++ )
     {
         unsigned int channelTypeID = bvhMotion->jointHierarchy[jID].channelType[channelNumber];
         unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,fID,channelTypeID);

         fprintf(stderr,"f[%u].%s.%s(%u)=%0.2f " ,
                 fID,
                 bvhMotion->jointHierarchy[jID].jointName,
                 channelNames[channelTypeID],
                 mID,
                 bvh_getMotionValue(bvhMotion,mID)
                 );
     }
    }
   fprintf(stderr,"\n\n");
  }
  fprintf(stderr,"_______________________________________________\n");
}



/*
int bvh_InternalCheck(struct BVH_MotionCapture * bvhMotion)
{
    unsigned int mID=0,jID=0,fID=0,cType=0;
    for (mID=0; mID<10000; mID++)
    {
       // struct BVH_MotionCapture * bvhMotion
       //TODO:
    }
}*/


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


int bvh_free(struct BVH_MotionCapture * bvhMotion)
{
  if ( bvhMotion==0 ) { return 0; }
  if ( bvhMotion->motionValues!= 0 ) { free(bvhMotion->motionValues); }
  return 1;
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


float bvh_getJointChannelAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID)
{
   if (bvhMotion==0) { return 0.0; }
   if (bvhMotion->jointHierarchySize<=jID) { return 0.0; }

   unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,fID,channelTypeID);

   if (mID>=bvhMotion->motionValuesSize)
   {
     fprintf(stderr,RED "bvh_getJointChannelAtFrame overflowed..\n" NORMAL);
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

float bvh_getMotionValue(struct BVH_MotionCapture * bvhMotion , unsigned int mID)
{
  return bvhMotion->motionValues[mID];
}




int bhv_jointHasParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID )
{
 if (jID>bvhMotion->jointHierarchySize) { return 0; }
 return (!bvhMotion->jointHierarchy[jID].isRoot);
}

int bhv_jointHasRotation(struct BVH_MotionCapture * bvhMotion , BVHJointID jID)
{
 if (jID>bvhMotion->jointHierarchySize) { return 0; }
 return (
          (bvhMotion->jointHierarchy[jID].loadedChannels>0) &&
          (bvhMotion->jointHierarchy[jID].channelRotationOrder!=0)
        );
}

