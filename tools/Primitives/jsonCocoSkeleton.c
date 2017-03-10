#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "skeleton.h"

#include "jsonCocoSkeleton.h"
#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/TrajectoryParser/InputParser_C.h"


int parseJsonCOCOSkeleton(const char * filename , struct skeletonCOCO * skel)
{
  fprintf(stderr,"Running COCO 2D skeleton \n");

//  char * line = NULL;
//  size_t len = 0;
  ssize_t read;

  FILE * fp = fopen(filename,"r");
  if (fp!=0)
  {
   struct InputParserC * ipc = InputParser_Create(1024,4);

   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,'[');
   InputParser_SetDelimeter(ipc,2,',');
   InputParser_SetDelimeter(ipc,3,']');
   /*
  {
"version":0.1,
"bodies":[
{
"joints":[365.493,70.4215,0.850086,344.691,110.847,0.832403,297.889,108.237,0.750012,255.007,168.241,0.854884,245.858,243.917,0.786749,388.92,114.794,0.758621,431.887,165.642,0.823047,443.558,226.957,0.550564,318.695,226.98,0.577757,395.419,255.682,0.799128,429.248,401.768,0.629969,375.954,219.153,0.669652,461.81,245.18,0.821205,569.769,330.016,0.447838,352.521,58.7188,0.873933,370.739,61.288,0.894234,329.126,65.2255,0.847911,0,0,0]
}]
}
*/
    char str[512];
    char * line = NULL;
    size_t len = 0;


    while ((read = getline(&line, &len, fp)) != -1)
    {
       //printf("Retrieved line of length %zu :\n", read);
       //printf("%s", line);

      if (strstr(line,"joints")!=0)
      {
       //printf("%s", line);
       int i=0;
       int num = InputParser_SeperateWords(ipc,line,1);


       //printf("%u parameters\n", num);
       if (num+1<COCO_PARTS*3)
       {
         fprintf(stderr,"Not enough joints provided for COCO format ( %u / %u )..\n",num+1 , COCO_PARTS*3);
       } else
       {
       for (i=0; i<COCO_PARTS; i++)
       {
        InputParser_GetWord(ipc,i,str,512);
        //printf("Joint %u ( %s ) ",i,COCOBodyNames[i]);
        skel->jointAccuracy[i] = InputParser_GetWordFloat(ipc,1+i*3+2);
        skel->joint2D[i].x     = InputParser_GetWordFloat(ipc,1+i*3+0);
        skel->joint2D[i].y     = InputParser_GetWordFloat(ipc,1+i*3+1);

        //printf("Pos ( x=%0.2f,y=%0.2f ) Precision %0.2f \n",skel->joint2D[i].x,skel->joint2D[i].y,skel->jointAccuracy[i]);
       }
       }
       //    doSkeletonConversions( &skel );
       //    printJointField ( &skel );
      }
    }
    fclose(fp);
    if (line) { free(line); }
    return 1;
  }

  return 0;
}
