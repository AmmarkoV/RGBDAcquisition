#include <stdio.h>
#include "bvh_loader.h"
#include "../TrajectoryParser/InputParser_C.h"

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

int parseHierarchy(unsigned int previousNode)
{
  return 0;
}

int readBVHHeader(struct BVH_MotionCapture * bvhMotion , FILE * fd )
{
  bvhMotion->numberOfValuesPerFrame = 0;//57;

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
    unsigned int channelNumber=0;
    unsigned int hierarchyLevel=0;
    unsigned int channels[8]={0};
    char * line = NULL;
    size_t len = 0;

    while ((read = getline(&line, &len, fd)) != -1)
    {
       if (strcmp(line,"}\n")==0) { break; }
       printf("Retrieved line of length %zu :\n", read);
       printf("%s", line);
       int num = InputParser_SeperateWords(ipc,line,1);


      if (num>0)
      { //We have content..
       if (!atHeaderSection)
       {
          if (InputParser_WordCompareAuto(ipc,0,"HIERARCHY"))  { atHeaderSection=1; }
       } else
       {
         int numB = InputParser_SeperateWords(ipcB,line,1);
         if (numB>0)
         {
         if (InputParser_WordCompareAuto(ipcB,0,"JOINT"))      {
                                                                  fprintf(stderr,"-J-");
                                                                  //Store new Joint
                                                                  InputParser_GetWord(ipcB,1, bvhMotion->jointHierarchy[bvhMotion->jointHierarchySize].jointName ,512);
                                                                  fprintf(stderr,"-%s-",bvhMotion->jointHierarchy[bvhMotion->jointHierarchySize].jointName);

                                                                  ++bvhMotion->jointHierarchySize;
                                                               } else
         if (InputParser_WordCompareAuto(ipcB,0,"End"))        {
                                                                 fprintf(stderr,"-E-");
                                                                 if (InputParser_WordCompareAuto(ipcB,1,"Site"))
                                                                  {
                                                                    fprintf(stderr,"-S-");

                                                                    snprintf(bvhMotion->jointHierarchy[bvhMotion->jointHierarchySize].jointName,512,"End Site");
                                                                    ++bvhMotion->jointHierarchySize;
                                                                  }
                                                               } else
         if (InputParser_WordCompareAuto(ipcB,0,"CHANNELS"))   {
                                                                 fprintf(stderr,"-C-");
                                                                 channelNumber=InputParser_GetWordInt(ipcB,1);
                                                                 bvhMotion->numberOfValuesPerFrame += channelNumber;
                                                                 fprintf(stderr,"-%u-",channelNumber);
                                                                 for (i=0; i<channelNumber; i++)
                                                                     {
                                                                       channels[i]=enumerateInputParserChannel(ipcB,2+i);
                                                                       fprintf(stderr,"-%u-",channels[i]);
                                                                     }
                                                               } else
         if (InputParser_WordCompareAuto(ipcB,0,"OFFSET"))     {
                                                                 fprintf(stderr,"-O-");
                                                                 bvhMotion->jointHierarchy[bvhMotion->jointHierarchySize].offset[0]=InputParser_GetWordFloat(ipcB,1);
                                                                 bvhMotion->jointHierarchy[bvhMotion->jointHierarchySize].offset[1]=InputParser_GetWordFloat(ipcB,2);
                                                                 bvhMotion->jointHierarchy[bvhMotion->jointHierarchySize].offset[2]=InputParser_GetWordFloat(ipcB,3);
                                                               } else
         if (InputParser_WordCompareAuto(ipcB,0,"{"))          {
                                                                  fprintf(stderr,"-{-");
                                                                  ++hierarchyLevel;
                                                               } else
         if (InputParser_WordCompareAuto(ipcB,0,"}"))          {
                                                                  fprintf(stderr,"-}-");
                                                                  if (hierarchyLevel>0)
                                                                  {
                                                                    --hierarchyLevel;
                                                                  } else
                                                                  {
                                                                    fprintf(stderr,"Erroneous BVH hierarchy..\n");
                                                                  }
                                                               }
          else
         {
            //Unexpected input..
         }

         } // We have header content
       } // We are at header section
      } //We have content
    } //We have line input from file

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
         if (InputParser_WordCompareAuto(ipc,0,"Frames"))      { bvhMotion->numberOfFrames = InputParser_GetWordInt(ipc,1); } else
         if (InputParser_WordCompareAuto(ipc,0,"Frame Time"))  { bvhMotion->frameTime = InputParser_GetWordFloat(ipc,1); }      else
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

int loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion)
{
  int successfullRead=0;
  FILE *fd=0;
  fd = fopen(filename,"r");
  if (fd!=0)
    {
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
