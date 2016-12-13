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


int compareSkeletons(  struct skeletonHuman * skelA , struct skeletonHuman * skelB)
{
 unsigned int i=0;
 for (i=0; i<HUMAN_SKELETON_PARTS-1; i++)
  {
    if ( (!skelA->active[i]) && (!skelB->active[i]) )
    {
      fprintf(stderr,"skel a AND b inactive(%s and %s)\n",smartBodyNames[i],tgbtNames[i]);
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

    if ( ( skelA->active[i] ) && (skelB->active[i]) )
    {
      float dis = distance3D(
                                &skelA->joint[i].x , &skelA->joint[i].y , &skelA->joint[i].z ,
                                &skelB->joint[i].x , &skelB->joint[i].y , &skelB->joint[i].z
                            );

      fprintf(stderr,"Distance(%s,%0.2f)\n",smartBodyNames[i],dis );
    }
 }
}


int main(int argc, char *argv[])
{

  struct skeletonHuman skelTRIBT={0};
  struct skeletonHuman skelFUBGT={0};


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
         if (InputParser_GetWordInt(ipc,1)>0)
         {
          printf("new frame   \n");
            compareSkeletons( &skelTRIBT , &skelFUBGT );
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
           for (i=0; i<HUMAN_SKELETON_PARTS-1; i++)
           {
            skelTRIBT.active[i]=0;
            //fprintf(stderr,"Comparing `%s` with `%s` \n",smartBodyNames[i],joint);
            if (strcmp(smartBodyNames[i],joint)==0 )
            {
              found=1;
              skelTRIBT.active[i]=1;
              skelTRIBT.joint[i].x =  InputParser_GetWordFloat(ipc,4);
              skelTRIBT.joint[i].y =  InputParser_GetWordFloat(ipc,5);
              skelTRIBT.joint[i].z =  InputParser_GetWordFloat(ipc,6);
            }
           }

          } else
          if (strcmp(tracker,"FUBGT")==0)
          {
           for (i=0; i<HUMAN_SKELETON_PARTS-1; i++)
           {
            skelFUBGT.active[i]=0;
            //fprintf(stderr,"Comparing `%s` with `%s` \n",humanSkeletonJointNames[i],joint);
            if (strcmp(humanSkeletonJointNames[i],joint)==0 )
            {
              found=1;
              skelFUBGT.active[i]=1;
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
