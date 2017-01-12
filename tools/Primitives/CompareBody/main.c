#include <stdio.h>
#include <stdlib.h>
#include "InputParser_C.h"
#include "../skeleton.h"
#include <math.h>

#define MAX_PATH 512

float distance3D(
                  float * ptA_X , float * ptA_Y , float * ptA_Z ,
                  float * ptB_X , float * ptB_Y , float * ptB_Z
                )
{
   float sqdiffX = (*ptA_X - *ptB_X);   sqdiffX = sqdiffX * sqdiffX;
   float sqdiffY = (*ptA_Y - *ptB_Y);   sqdiffY = sqdiffY * sqdiffY;
   float sqdiffZ = (*ptA_Z - *ptB_Z);   sqdiffZ = sqdiffZ * sqdiffZ;

   return sqrt( (sqdiffX + sqdiffY + sqdiffZ) );
}

int printPartsOfSkeletonsInactive(unsigned int frameNum ,  struct skeletonHuman * skelA , struct skeletonHuman * skelB)
{
  unsigned int i;
  unsigned int activePairs=0;
 //Error announcement
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
  {
    if ( (skelA->active[i]) && (skelB->active[i]) )
    {
     ++activePairs;
    } else
    if ( (!skelA->active[i]) && (!skelB->active[i]) )
    {
      fprintf(stderr,"compareSkeletons(%u) skel a AND b inactive(%s and %s , %u [ %u %u ]  )\n",frameNum,smartBodyNames[i],tgbtNames[i] , i , skelA->active[i] , skelB->active[i] );
    } else
    {
      if(!skelB->active[i])
       {
        fprintf(stderr,"compareSkeletons(%u) skel b inactive(%s)\n",frameNum,tgbtNames[i]);
       } else
      if(!skelA->active[i])
       {
        fprintf(stderr,"compareSkeletons(%u) skel a inactive(%s)\n",frameNum,smartBodyNames[i]);
       }
    }
  }

 return activePairs;
}


int compareSkeletons( unsigned int frameNum ,  struct skeletonHuman * skelA , struct skeletonHuman * skelB)
{
 unsigned int i=0;
 unsigned int activePairs=0;
 float totalDistance = 0;

 activePairs=printPartsOfSkeletonsInactive( frameNum ,  skelA , skelB);
 if (!activePairs) { return 0; }

 //Everything same line
 fprintf(stdout,"PerJointDistance ");
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
  {
    if ( ( skelA->active[i] ) && (skelB->active[i]) )
    {
      if
         (
          ( (skelA->joint[i].x!=0) && (skelA->joint[i].y) && (skelA->joint[i].z) ) ||
          ( (skelB->joint[i].x!=0) && (skelB->joint[i].y) && (skelB->joint[i].z) )
         )
      {
       float dis = distance3D(
                                 &skelA->joint[i].x , &skelA->joint[i].y , &skelA->joint[i].z ,
                                 &skelB->joint[i].x , &skelB->joint[i].y , &skelB->joint[i].z
                             );
        totalDistance += dis;
        fprintf(stdout,"%0.2f " , dis );
      } else
      { fprintf(stdout,"0.0 "); }
    } else
    { fprintf(stdout,"0.0 "); }
 }
 fprintf(stdout,"\n");


 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
  {
    if ( ( skelA->active[i] ) && (skelB->active[i]) )
    {
      if
         (
          ( (skelA->joint[i].x!=0) && (skelA->joint[i].y!=0) && (skelA->joint[i].z!=0) ) ||
          ( (skelB->joint[i].x!=0) && (skelB->joint[i].y!=0) && (skelB->joint[i].z!=0) )
         )
      {
       float dis = distance3D(
                                 &skelA->joint[i].x , &skelA->joint[i].y , &skelA->joint[i].z ,
                                 &skelB->joint[i].x , &skelB->joint[i].y , &skelB->joint[i].z
                             );
        totalDistance += dis;
        fprintf(stdout,"JointDistance %s %u %0.2f \n",smartBodyNames[i],frameNum , dis );
      }
    }
 }

 //Test Joint Lengths
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
  {
    if ( ( skelA->active[i] ) && (skelB->active[i]) )
    {
      if
         (
          ( (skelA->joint[i].x!=0) && (skelA->joint[i].y!=0) && (skelA->joint[i].z!=0) ) ||
          ( (skelB->joint[i].x!=0) && (skelB->joint[i].y!=0) && (skelB->joint[i].z!=0) )
         )
      {
        float j1L = skeleton3DGetJointLength(skelA,i);
        float j2L = skeleton3DGetJointLength(skelB,i);

        float jLDis = (float) j1L-j2L;
        if (jLDis<0) { jLDis= -1*jLDis; }

        fprintf(stdout,"JointLengthDifference %s %u %0.2f %0.2f %0.2f \n",smartBodyNames[i],frameNum , jLDis , j1L , j2L );
      }
    }
 }




 fprintf(stdout,"TotalDistance %u %0.2f\n",frameNum , totalDistance);
 return 1;
}



int readLine(struct InputParserC * ipc  , char * line , unsigned int * frameNumber ,  struct skeletonHuman * skelTRIBT , struct skeletonHuman * skelFUBGT )
{
 //printf("Retrieved line of length %zu :\n", read);
  char tag[MAX_PATH]={0};
  char tracker[MAX_PATH]={0};
  char joint[MAX_PATH]={0};


  InputParser_SeperateWords(ipc,line,0);
  InputParser_GetWord(ipc,0,tag,MAX_PATH);

  if (strcmp(tag,"FRAME")==0)
       {
         *frameNumber = InputParser_GetWordInt(ipc,1);
         if (*frameNumber>0)
         {
          printf("new frame %u  \n", *frameNumber);
          return 1;
         }
       }
         else
  if (strcmp(tag,"JOINT")==0)
       {
          unsigned int i,found=0;
          //printf("%s", line);
          InputParser_GetWord(ipc,1,tracker,MAX_PATH);
          InputParser_GetWord(ipc,3,joint,MAX_PATH);


          if (strcmp(tracker,"TRIBT")==0)
          {
           for (i=0; i<HUMAN_SKELETON_PARTS; i++)
           {
            //fprintf(stderr,"Comparing `%s` with `%s` \n",smartBodyNames[i],joint);
            if (strcasecmp(smartBodyNames[i],joint)==0 )
            {
              found=1;
              skelTRIBT->active[i]=1;
              //fprintf(stderr,"frame(%u)  skelTRIBT.active[%u]=%u;\n",InputParser_GetWordInt(ipc,2),i,skelTRIBT->active[i]);
              skelTRIBT->joint[i].x =  InputParser_GetWordFloat(ipc,4);
              skelTRIBT->joint[i].y =  InputParser_GetWordFloat(ipc,5);
              skelTRIBT->joint[i].z =  InputParser_GetWordFloat(ipc,6);
            }
           }
          } else
          if (strcmp(tracker,"FUBGT")==0)
          {
           for (i=0; i<HUMAN_SKELETON_PARTS; i++)
           {
            //fprintf(stderr,"Comparing `%s` with `%s` \n",humanSkeletonJointNames[i],joint);
            if (strcasecmp(humanSkeletonJointNames[i],joint)==0 )
            {
              found=1;
              skelFUBGT->active[i]=1;
              //fprintf(stderr,"frame(%u) skelFUBGT.active[%u]=%u;\n",InputParser_GetWordInt(ipc,2),i,skelFUBGT->active[i]);
              skelFUBGT->joint[i].x =  InputParser_GetWordFloat(ipc,4);
              skelFUBGT->joint[i].y =  InputParser_GetWordFloat(ipc,5);
              skelFUBGT->joint[i].z =  InputParser_GetWordFloat(ipc,6);
            }
           }
          } else
          {
            fprintf(stderr,"Unknown tracker (%s) \n",tracker);
          }

           if (!found)
             {
               fprintf(stderr,"Could not find `%s` \n",joint);
             }

       }
  return 0;
}





int compare1OutputOf2Trackers(char * trackerOutputFilename)
{
  struct skeletonHuman skelTRIBT={0};
  struct skeletonHuman skelFUBGT={0};

  unsigned int frameNumber=0;

  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  FILE * fp = fopen(trackerOutputFilename,"r");
  if (fp!=0)
  {
    struct InputParserC * ipc = InputParser_Create(2048,4);

    char * line = NULL;
    size_t len = 0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
      if ( readLine(ipc , line , &frameNumber , &skelTRIBT , &skelFUBGT ) )
      {
        //got a match
        compareSkeletons( frameNumber , &skelTRIBT , &skelFUBGT );
       //clear active joints for next frame
       unsigned int i=0;
            for (i=0; i<HUMAN_SKELETON_PARTS; i++)
             {
                 skelTRIBT.active[i]=0;
                 skelFUBGT.active[i]=0;
             }
      }
    }

    InputParser_Destroy(ipc);
    fclose(fp);
    if (line) { free(line); }
  }
return 1;
}





int compare2Skeletons( unsigned int frameNum ,
                       struct skeletonHuman * skelGroundA , struct skeletonHuman * skelGroundB ,
                       struct skeletonHuman * skelA , struct skeletonHuman * skelB)
{
 unsigned int i=0;
 unsigned int activePairs=0;
 float totalDistance = 0;

 activePairs=printPartsOfSkeletonsInactive( frameNum ,  skelGroundA , skelGroundB);
 if (!activePairs) { return 0; }

 activePairs=printPartsOfSkeletonsInactive( frameNum ,  skelA , skelB);
 if (!activePairs) { return 0; }

 //Everything same line
 fprintf(stdout,"PerJointDistanceGround ");
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
  {
    if ( ( skelA->active[i] ) && (skelGroundA->active[i]) )
    {
      if
         (
          ( (skelA->joint[i].x!=0) && (skelA->joint[i].y!=0) && (skelA->joint[i].z!=0) ) ||
          ( (skelGroundA->joint[i].x!=0) && (skelGroundA->joint[i].y!=0) && (skelGroundA->joint[i].z!=0) )
         )
      {
       float dis = distance3D(
                                 &skelA->joint[i].x , &skelA->joint[i].y , &skelA->joint[i].z ,
                                 &skelGroundA->joint[i].x , &skelGroundA->joint[i].y , &skelGroundA->joint[i].z
                             );
        totalDistance += dis;
        fprintf(stdout,"%0.2f " , dis );
      } else
      { fprintf(stdout,"0.0 "); }
    } else
    { fprintf(stdout,"0.0 "); }
 }
 fprintf(stdout,"\n");


 fprintf(stdout,"TotalDistance %u %0.2f\n",frameNum , totalDistance);
 return 1;
}






int compare2OutputOf2Trackers(char * trackerOutputFilenameGround , char * trackerOutputFilename)
{
  struct skeletonHuman skelTRIBTGround={0};
  struct skeletonHuman skelFUBGTGround={0};

  struct skeletonHuman skelTRIBT={0};
  struct skeletonHuman skelFUBGT={0};

  unsigned int frameNumber=0;
  unsigned int readFromGroundFile=1;

  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  FILE * fp = fopen(trackerOutputFilename,"r");
  FILE * fpGround = fopen(trackerOutputFilenameGround,"r");
  if ( (fp!=0) && (fpGround!=0) )
  {
    struct InputParserC * ipc = InputParser_Create(2048,4);

    char * line = NULL;
    size_t len = 0;

    char * lineGround = NULL;

    while ((read = getline(&line, &len, fp)) != -1)
    {
      if (readFromGroundFile)
      {
      while ((read = getline(&lineGround, &len, fpGround)) != -1)
       {
        if ( readLine(ipc , lineGround , &frameNumber , &skelTRIBTGround , &skelFUBGTGround ) )
        {
         readFromGroundFile=0;
         break;
        }
       }
      }

      if ( readLine(ipc , line , &frameNumber , &skelTRIBT , &skelFUBGT ) )
      {
        //got a match
        compare2Skeletons( frameNumber , &skelTRIBTGround , &skelFUBGTGround , &skelTRIBT , &skelFUBGT );
        readFromGroundFile=1;
       //clear active joints for next frame
       unsigned int i=0;
            for (i=0; i<HUMAN_SKELETON_PARTS; i++)
             {
                 skelTRIBT.active[i]=0;
                 skelFUBGT.active[i]=0;
                 skelTRIBTGround.active[i]=0;
                 skelFUBGTGround.active[i]=0;
             }
      }
    }

    InputParser_Destroy(ipc);
    fclose(fp);
    if (line) { free(line); }
  }
return 1;
}









int main(int argc, char *argv[])
{

  if (argv<2)
     { fprintf(stderr,"./CompareBody --single one.scene  OR --ground ground.scene two.scene\n"); return 0; }

  if (strcmp(argv[1],"--single")==0)  {  compare1OutputOf2Trackers(argv[2]);         } else
  if (strcmp(argv[1],"--ground")==0)  {  compare2OutputOf2Trackers(argv[2],argv[3]); }

  return 0;
}

