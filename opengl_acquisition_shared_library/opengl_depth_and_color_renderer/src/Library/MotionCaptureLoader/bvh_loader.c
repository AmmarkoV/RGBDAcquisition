#include <stdio.h>
#include "bvh_loader.h"
#include "../TrajectoryParser/InputParser_C.h"

//http://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html?fbclid=IwAR0BopXj4Kft_RAEE41VLblkkPGHVF8-mon3xSCBMZueRtyb9LCSZDZhXPA

int readBVHMotion(struct BVH_MotionCapture * bvhMotion , FILE * fd )
{

  ssize_t read;
  unsigned int frameNumber =0;

  if (fd!=0)
  {
   struct InputParserC * ipc = InputParser_Create(1024,4);

   InputParser_SetDelimeter(ipc,0,':');
   InputParser_SetDelimeter(ipc,1,'[');
   InputParser_SetDelimeter(ipc,2,',');
   InputParser_SetDelimeter(ipc,3,']');

   /*
MOTION
Frames:    2
Frame Time: 0.033333
*/
    char str[512];
    char * line = NULL;
    size_t len = 0;


    while ((read = getline(&line, &len, fd)) != -1)
    {
       printf("Retrieved line of length %zu :\n", read);
       printf("%s", line);

      if (strstr(line,"Frames:")!=0)
      {
       //printf("%s", line);
       int i=0;
       int num = InputParser_SeperateWords(ipc,line,1);
      }

    }
  }
}


int readBVHHeader(struct BVH_MotionCapture * bvhMotion , FILE * fd )
{
  fprintf(stderr,"Skipping header for now..\n");
}


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
