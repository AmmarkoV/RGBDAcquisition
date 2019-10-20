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


#include "edit/bvh_cut_paste.h"
/*
//THIS IS NOT USED ANYWHERE
double bvh_constrainAngleCentered180(double angle)
{
   angle = fmod(angle,360.0);
   if (angle<0.0)
     { angle+=360.0; }
   return angle;
}
*/


// We have circles A , B and C and we are trying to map circles A and B to circle C
// Because neural networks get confused when coordinates jump from 0 to 360
//
//                                -360 A 0
//
//                                   0 B 360
//
//                                -180 C 180
//
//
//      -270A . 90B . -90C             *                 90C  270B  -90A
//
//
//                                  -1 C 1
//
//                                 179 B 181
//
//                                -181 C -179
//
//
//We want to add 180 degrees to the model so 0 is oriented towards us..!
double bvh_constrainAngleCentered0(double angle,unsigned int flipOrientation)
{
    double angleFrom_minus360_to_plus360;
    double angleRotated = angle+180;

     if (angleRotated<0.0)
     {
       angleFrom_minus360_to_plus360 = (-1*fmod(-1*(angleRotated),360.0))+180;
     } else
     {
       angleFrom_minus360_to_plus360 = (fmod((angleRotated),360.0))-180;
     }

    //If we want to flip orientation we just add or subtract 180 depending on the case
    //To retrieve correct orientatiation we do the opposite
    if (flipOrientation)
    {
      if (angleFrom_minus360_to_plus360<0.0) { angleFrom_minus360_to_plus360+=180.0; } else
      if (angleFrom_minus360_to_plus360>0.0) { angleFrom_minus360_to_plus360-=180.0; } else
                                             { angleFrom_minus360_to_plus360=180.0;  }
    }

   return angleFrom_minus360_to_plus360;
}



double bvh_RemapAngleCentered0(double angle, unsigned int constrainOrientation)
{
   double angleShifted = angle; 
   //We want to add 180 degrees to the model so 0 is oriented towards us..! 
   switch (constrainOrientation)
   {
      case BVH_ENFORCE_NO_ORIENTATION :                          return bvh_constrainAngleCentered0(angleShifted,0); break;      
      case BVH_ENFORCE_FRONT_ORIENTATION :                       return bvh_constrainAngleCentered0(angleShifted,0); break;      
      case BVH_ENFORCE_BACK_ORIENTATION :                        return bvh_constrainAngleCentered0(angleShifted,1); break;      
      case BVH_ENFORCE_LEFT_ORIENTATION :    angleShifted+=90.0; return bvh_constrainAngleCentered0(angleShifted,0); break;      
      case BVH_ENFORCE_RIGHT_ORIENTATION :   angleShifted+=90.0; return bvh_constrainAngleCentered0(angleShifted,1); break; 
   };

   fprintf(stderr,"Did not change angle, due to incorrect BVH_ENFORCE_XX constrain..\n");
  return angle;
}



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

  int debug=0;

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
               if (debug) fprintf(stderr,"-R-");
               //Store new ROOT Joint Name
               InputParser_GetWord(
                                    ipcB,1,
                                    bvhMotion->jointHierarchy[jNum].jointName ,
                                    MAX_BVH_JOINT_NAME
                                  );
               if (debug) fprintf(stderr,"-%s-",bvhMotion->jointHierarchy[jNum].jointName);
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
               if (debug) fprintf(stderr,"-J-");
               //Store new Joint Name
               InputParser_GetWord(
                                    ipcB,1,
                                    bvhMotion->jointHierarchy[jNum].jointName ,
                                    MAX_BVH_JOINT_NAME
                                  );
               if (debug) fprintf(stderr,"-%s-",bvhMotion->jointHierarchy[jNum].jointName);
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
              if (debug) fprintf(stderr,"-E-");
              if (InputParser_WordCompareAuto(ipcB,1,"Site"))
                   {
                    if (debug) fprintf(stderr,"-S-");
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
              if (debug) fprintf(stderr,"-C");

              //Keep as shorthand..
              unsigned int parentID=bvhMotion->jointHierarchy[currentJoint].parentJoint;

              //Read number of Channels
              unsigned int loadedChannels = InputParser_GetWordInt(ipcB,1);
              bvhMotion->jointHierarchy[currentJoint].loadedChannels=loadedChannels;
              if (debug) fprintf(stderr,"(%u)-",loadedChannels);

              //First wipe channels to make sure they are clean
              memset(bvhMotion->jointHierarchy[currentJoint].channelType,0,sizeof(char) * BVH_VALID_CHANNEL_NAMES);

              if (debug) fprintf(stderr,"\nJOINT %u (%s) : ",currentJoint,bvhMotion->jointHierarchy[currentJoint].jointName);

              //Now to store the channel labels
              unsigned int cL=0; //Channel To Load
              for (cL=0; cL<loadedChannels; cL++)
                  {
                   //For each declared channel we need to enumerate the label to a value
                   unsigned int thisChannelID = enumerateInputParserChannel(ipcB,2+cL);

                   bvhMotion->jointHierarchy[currentJoint].channelType[cL]=thisChannelID;

                   if (debug) fprintf(stderr,"#%u %s=%u ",cL,channelNames[thisChannelID],bvhMotion->numberOfValuesPerFrame);

                   //Update jointToMotion Lookup Table..
                   bvhMotion->jointToMotionLookup[currentJoint].channelIDMotionOffset[thisChannelID] = bvhMotion->numberOfValuesPerFrame;

                   //Update motionToJoint Lookup Table..
                   bvhMotion->motionToJointLookup[bvhMotion->numberOfValuesPerFrame].channelID = thisChannelID;
                   bvhMotion->motionToJointLookup[bvhMotion->numberOfValuesPerFrame].jointID   = currentJoint;
                   bvhMotion->motionToJointLookup[bvhMotion->numberOfValuesPerFrame].parentID  = parentID;

                   ++bvhMotion->numberOfValuesPerFrame;
                  }
                if (debug) fprintf(stderr,"\n");

               bvhMotion->jointHierarchy[currentJoint].channelRotationOrder = enumerateChannelOrder(bvhMotion,currentJoint);
              //Done
              } else
         if (InputParser_WordCompareAuto(ipcB,0,"OFFSET"))
             {//Reached something like |OFFSET	 3.91	 0.00	 0.00|
              if (debug) fprintf(stderr,"-O-");

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
              if (debug) fprintf(stderr,"-{%u-",hierarchyLevel);
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
              if (debug) fprintf(stderr,"-%u}-",hierarchyLevel);
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
int bvh_loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion, float scaleWorld)
{
 bvhMotion->scaleWorld=scaleWorld;
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

   double buffer = (double) mc->motionValues[mID+3];
   buffer = bvh_RemapAngleCentered0(buffer,0);
   mc->motionValues[mID+3] = (float) buffer;

   buffer = (double) mc->motionValues[mID+4];
   buffer = bvh_RemapAngleCentered0(buffer,constrainOrientation);
   mc->motionValues[mID+4] = (float) buffer;

   buffer = (double) mc->motionValues[mID+5];
   buffer = bvh_RemapAngleCentered0(buffer,0);
   mc->motionValues[mID+5] = (float) buffer;
  }
 return 1;
}


int bvh_testConstrainRotations()
{
  /*  
  fprintf(stderr,"Testing bvh_rotation constraint\n");
  unsigned int i=0;
  double angle = -720;
  for (i=0; i<1440; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Centered at 0:%0.2f | Flipped at 0:%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_constrainAngleCentered0(angle,0),
    bvh_constrainAngleCentered0(angle,1)
    //bvh_constrainAngleCentered180(angle)
    );
    angle=angle+1.0;
  }
  
  */
  

  fprintf(stderr,"Testing bvh_rotation front constraint\n");
  unsigned int i=0;
  double angle = -45;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Front Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_FRONT_ORIENTATION)
    );
    angle=angle+1.0;
  } 
   

  fprintf(stderr,"Testing bvh_rotation back constraint\n");
  i=0;
  angle = 135;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Back Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_BACK_ORIENTATION)
    );
    angle=angle+1.0;
  } 
   

  fprintf(stderr,"Testing bvh_rotation right constraint\n");
  i=0;
  angle = 45;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Right Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_RIGHT_ORIENTATION)
    );
    angle=angle+1.0;
  } 


  fprintf(stderr,"Testing bvh_rotation left constraint\n");
  i=0;
  angle = -135;
  for (i=0; i<90; i++)
  {
    fprintf(stderr,"| Angle:%0.2f | Left Centered at 0 :%0.2f\n", //| Centered at 180:%0.2f
    angle,
    bvh_RemapAngleCentered0(angle,BVH_ENFORCE_LEFT_ORIENTATION)
    );
    angle=angle+1.0;
  } 
   
  
 return 0;
}

int bvh_InterpolateMotion(
                           struct BVH_MotionCapture * mc
                         )
{
  if (mc==0) { return 0; }

  float * newMotionValues = (float*) malloc(sizeof(float) * mc->motionValuesSize * 2 );
  unsigned int i,z,target=0;
  for (i=0; i<mc->numberOfFrames-1; i++)
    {
      //First copy frame
      for (z=0; z<mc->numberOfValuesPerFrame; z++)
        { newMotionValues[target*mc->numberOfValuesPerFrame + z] = mc->motionValues[(i)*mc->numberOfValuesPerFrame + z]; }
      target++;
      //Then add an interpolated frame
      for (z=0; z<mc->numberOfValuesPerFrame; z++)
        { newMotionValues[target*mc->numberOfValuesPerFrame + z] = ( mc->motionValues[i*mc->numberOfValuesPerFrame + z] + mc->motionValues[(i+1)*mc->numberOfValuesPerFrame + z] ) / 2;  }
      target++;
    }

 //Copy last two frames
 i=mc->numberOfFrames-1;
 for (z=0; z<mc->numberOfValuesPerFrame; z++)
      { newMotionValues[target*mc->numberOfValuesPerFrame + z] = mc->motionValues[(i)*mc->numberOfValuesPerFrame + z]; }
  target++;
 for (z=0; z<mc->numberOfValuesPerFrame; z++)
      { newMotionValues[target*mc->numberOfValuesPerFrame + z] = mc->motionValues[(i)*mc->numberOfValuesPerFrame + z]; }


  float * oldMotionValues = mc->motionValues;
  mc->frameTime=mc->frameTime/2;
  mc->numberOfFrames=mc->numberOfFrames*2;
  mc->numberOfFramesEncountered=mc->numberOfFrames;
  mc->motionValuesSize=mc->motionValuesSize*2;
  mc->motionValues = newMotionValues;
  free(oldMotionValues);

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


int bvh_getJointIDFromJointNameNocase(
                                      struct BVH_MotionCapture * bvhMotion ,
                                      const char * jointName,
                                      BVHJointID * jID
                                     )
{
  if (bvhMotion==0) { return 0; }
  if (strlen(jointName)>=MAX_BVH_JOINT_NAME)
     {
       fprintf(stderr,"bvh_getJointIDFromJointNameNocase failed because of very long joint names..");
       return 0;
     }

   char jointNameLowercase[MAX_BVH_JOINT_NAME+1]={0};
   snprintf(jointNameLowercase,MAX_BVH_JOINT_NAME,"%s",jointName);
   lowercase(jointNameLowercase);

   return bvh_getJointIDFromJointName(
                                      bvhMotion ,
                                      jointNameLowercase,
                                      jID
                                     );
}







int bvh_getRootJointID(
                       struct BVH_MotionCapture * bvhMotion ,
                       BVHJointID * jID
                      )
{
   if (bvhMotion==0) { return 0; }

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

int bvh_onlyAnimateGivenJoints(struct BVH_MotionCapture * bvhMotion,unsigned int numberOfArguments,char **argv)
{
    bvh_printBVH(bvhMotion);
    fprintf(stderr,"bvh_onlyAnimateGivenJoints with %u arguments\n",numberOfArguments);

    BVHJointID * activeJoints = (BVHJointID*) malloc(sizeof(BVHJointID) * numberOfArguments);
    memset(activeJoints,0,sizeof(BVHJointID) * numberOfArguments);

    char * successJoints = (char *) malloc(sizeof(char) * numberOfArguments);
    memset(successJoints,0,sizeof(char) * numberOfArguments);


    if ((activeJoints!=0) && (successJoints!=0))
    {

    for (int i=0; i<numberOfArguments; i++)
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


      unsigned int firstFrame=0,jointID=0;
      unsigned int mID_Initial,mID_Target;
      for (int frameID=0; frameID<bvhMotion->numberOfFramesEncountered; frameID++)
       {
         //fprintf(stderr,"FrameNumber %u\n",frameID);
         for (int mID=0; mID<bvhMotion->numberOfValuesPerFrame; mID++)
         {

             jointID = bvhMotion->motionToJointLookup[mID].jointID;
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


//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
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
float bvh_getJointChannelAtMotionBuffer(struct BVH_MotionCapture * bvhMotion, BVHJointID jID,float * motionBuffer, unsigned int channelTypeID)
{
   if (bvhMotion==0) { return 0.0; }
   if (bvhMotion->jointHierarchySize<=jID) { return 0.0; }

   unsigned int mID = bvh_resolveFrameAndJointAndChannelToMotionID(bvhMotion,jID,0,channelTypeID);

   if (mID>=bvhMotion->motionValuesSize)
   {
     fprintf(stderr,RED "bvh_getJointChannelAtMotionBuffer overflowed..\n" NORMAL);
     return 0.0;
   }

   return motionBuffer[mID];
}

float  bvh_getJointRotationXAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_X); }
float  bvh_getJointRotationYAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_Y); }
float  bvh_getJointRotationZAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_ROTATION_Z); }
float  bvh_getJointPositionXAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_POSITION_X); }
float  bvh_getJointPositionYAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_POSITION_Y); }
float  bvh_getJointPositionZAtMotionBuffer(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,float * motionBuffer) { return bvh_getJointChannelAtMotionBuffer(bvhMotion,jID,motionBuffer,BVH_POSITION_Z); }


int bhv_populatePosXYZRotXYZFromMotionBuffer(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , float * motionBuffer, float * data, unsigned int sizeOfData)
{
  if (data == 0) { return 0; }
  if (sizeOfData < sizeof(float)* 6) { return 0; }

  data[0]=bvh_getJointPositionXAtMotionBuffer(bvhMotion,jID,motionBuffer);
  data[1]=bvh_getJointPositionYAtMotionBuffer(bvhMotion,jID,motionBuffer);
  data[2]=bvh_getJointPositionZAtMotionBuffer(bvhMotion,jID,motionBuffer);
  data[3]=bvh_getJointRotationXAtMotionBuffer(bvhMotion,jID,motionBuffer);
  data[4]=bvh_getJointRotationYAtMotionBuffer(bvhMotion,jID,motionBuffer);
  data[5]=bvh_getJointRotationZAtMotionBuffer(bvhMotion,jID,motionBuffer);
  return 1;
}
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------
//------------------ ------------------ ------------------ ------------------ ------------------ ------------------ ------------------



float bvh_getMotionValue(struct BVH_MotionCapture * bvhMotion , unsigned int mID)
{
  return bvhMotion->motionValues[mID];
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
    fprintf(stderr,"level %u\n",bvhMotion->jointHierarchy[i].hierarchyLevel );
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

