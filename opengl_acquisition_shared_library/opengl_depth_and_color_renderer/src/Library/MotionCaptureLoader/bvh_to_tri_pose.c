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

    swhile  ((read = getline(&line, &len, fp)) != -1)
    {

       int num = InputParser_SeperateWords(ipc,line,1);

       if (InputParser_WordCompareAuto(ipc,0,"JOINT_ASSOCIATION"))
         {
         } else
       if (InputParser_WordCompareAuto(ipc,0,"JOINT_ROTATION"))
         {
         } else
       if (InputParser_WordCompareAuto(ipc,0,"JOINT_OFFSET"))
        {
        }
    }

    if (line) { free(line); }

    fclose(fp);
    InputParser_Destroy(ipc);
  }
return 0;
}













