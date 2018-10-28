#include <stdio.h>
#include <stdlib.h>
#include "bvh_to_tri_pose.h"

#include "../TrajectoryParser/InputParser_C.h"

int bvh_loadBVHToTRI(const char * filename , struct bvhToTRI * bvhtri)
{
  FILE * fp = fopen(filename,"r");
  if (fp!=0)
  {
   struct InputParserC * ipc = InputParser_Create(4096,7);
   InputParser_SetDelimeter(ipc,0,' ');
   InputParser_SetDelimeter(ipc,1,'(');
   InputParser_SetDelimeter(ipc,2,',');
   InputParser_SetDelimeter(ipc,3,')');
   InputParser_SetDelimeter(ipc,4,'\t');
   InputParser_SetDelimeter(ipc,5,10);
   InputParser_SetDelimeter(ipc,6,13);


    ssize_t read;
    char * line = NULL;
    size_t len = 0;

    char nameA[512]={0};
    char nameB[512]={0};

    char rotA[512]={0};
    char rotB[512]={0};
    char rotC[512]={0};

    float offsetA=0.0;
    float offsetB=0.0;
    float offsetC=0.0;


    while  ((read = getline(&line, &len, fp)) != -1)
    {
       int num = InputParser_SeperateWords(ipc,line,1);

       if (num>0)
       {
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_ASSOCIATION"))
         {
           InputParser_GetWord(ipc,1,nameA,512);
           InputParser_GetWord(ipc,2,nameB,512);
           fprintf(stderr,"Associated `%s` => `%s` \n",nameA,nameB);
         } else
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_ROTATION_ORDER"))
         {
           InputParser_GetWord(ipc,1,nameA,512);
           InputParser_GetWord(ipc,2,rotA,512);
           InputParser_GetWord(ipc,3,rotB,512);
           InputParser_GetWord(ipc,4,rotC,512);
           fprintf(stderr,"RotationOrder %s(%s,%s,%s)\n",nameA,rotA,rotB,rotC);
         } else
        if (InputParser_WordCompareAuto(ipc,0,"JOINT_OFFSET"))
        {
          InputParser_GetWord(ipc,1,nameA,512);
          offsetA = InputParser_GetWordFloat(ipc,2);
          offsetB = InputParser_GetWordFloat(ipc,3);
          offsetC = InputParser_GetWordFloat(ipc,4);
           fprintf(stderr,"Offset %s(%0.2f,%0.2f,%0.2f)\n",nameA,offsetA,offsetB,offsetC);
        }
       }
    }

    if (line) { free(line); }

    fclose(fp);
    InputParser_Destroy(ipc);
  }
return 0;
}













