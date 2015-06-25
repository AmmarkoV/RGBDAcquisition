#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../OGLRendererSandbox.h"


int main(int argc, char **argv)
{
   if (argc<4)
   {
     fprintf(stderr,"usage : Comparer path/To/FileA.scene path/To/FileB.scene numberOfFrames \n");
     return 0;
   }
   compareTrajectoryFiles("comparison.txt",argv[1],argv[2],atoi(argv[3]));

  return 0;
}
