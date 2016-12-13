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
   float sqdiffX = (*ptA_X - *ptB_X);
   sqdiffX = sqdiffX * sqdiffX;

   float sqdiffY = (*ptA_Y - *ptB_Y);
   sqdiffY = sqdiffY * sqdiffY;

   float sqdiffZ = (*ptA_Z - *ptB_Z);
   sqdiffZ = sqdiffZ * sqdiffZ;

   return sqrt( (sqdiffX + sqdiffY + sqdiffZ) );
}


int compareSkeletons( unsigned int frameNum ,  struct skeletonHuman * skelA , struct skeletonHuman * skelB)
{
 unsigned int i=0;
 unsigned int activePairs=0;
 float totalDistance = 0;

 //Error announcement
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
  {
    if ( (skelA->active[i]) && (skelB->active[i]) )
    {
     ++activePairs;
    } else
    if ( (!skelA->active[i]) && (!skelB->active[i]) )
    {
      fprintf(stderr,"skel a AND b inactive(%s and %s , %u [ %u %u ]  )\n",smartBodyNames[i],tgbtNames[i] , i , skelA->active[i] , skelB->active[i] );
    } else
    {
      if(!skelB->active[i])
       {
        fprintf(stderr,"skel b inactive(%s)\n",tgbtNames[i]);
       } else
      if(!skelA->active[i])
       {
        fprintf(stderr,"skel a inactive(%s)\n",smartBodyNames[i]);
       }
    }
  }

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
      {
       fprintf(stdout,"0.0 ");
      }
    } else
    {
      fprintf(stdout,"0.0 ");
    }
 }
 fprintf(stdout,"\n");








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
        fprintf(stdout,"JointDistance %s %u %0.2f \n",smartBodyNames[i],frameNum , dis );
      }
    }
 }

 fprintf(stdout,"TotalDistance %u %0.2f\n",frameNum , totalDistance);
 return 1;
}


int main(int argc, char *argv[])
{

  struct skeletonHuman skelTRIBT={0};
  struct skeletonHuman skelFUBGT={0};

  unsigned int frameNumber=0;
  char tag[MAX_PATH]={0};
  char tracker[MAX_PATH]={0};
  char joint[MAX_PATH]={0};

  char * line = NULL;
  size_t len = 0;
  ssize_t read;

  FILE * fp = fopen(argv[1],"r");
  if (fp!=0)
  {
    struct InputParserC * ipc = InputParser_Create(2048,4);

    char * line = NULL;
    size_t len = 0;

    while ((read = getline(&line, &len, fp)) != -1)
    {
       //printf("Retrieved line of length %zu :\n", read);

       InputParser_SeperateWords(ipc,line,0);


       InputParser_GetWord(ipc,0,tag,MAX_PATH);

       if (strcmp(tag,"FRAME")==0)
       {
         frameNumber = InputParser_GetWordInt(ipc,1);
         if (frameNumber>0)
         {
          printf("new frame %u  \n", frameNumber);
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
              skelTRIBT.active[i]=1;
              //fprintf(stderr," skelTRIBT.active[%u]=%u;\n",i,skelTRIBT.active[i]);
              skelTRIBT.joint[i].x =  InputParser_GetWordFloat(ipc,4);
              skelTRIBT.joint[i].y =  InputParser_GetWordFloat(ipc,5);
              skelTRIBT.joint[i].z =  InputParser_GetWordFloat(ipc,6);
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
              skelFUBGT.active[i]=1;
              //fprintf(stderr," skelFUBGT.active[%u]=%u;\n",i,skelFUBGT.active[i]);
              skelFUBGT.joint[i].x =  InputParser_GetWordFloat(ipc,4);
              skelFUBGT.joint[i].y =  InputParser_GetWordFloat(ipc,5);
              skelFUBGT.joint[i].z =  InputParser_GetWordFloat(ipc,6);
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
    }

    InputParser_Destroy(ipc);
    fclose(fp);
    if (line) { free(line); }

 }







    return 0;
}
