#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../../Library/OGLRendererSandbox.h"


int main(int argc, char **argv)
{
   if (argc<6)
   {
     fprintf(stderr,"usage : Comparer path/To/FileA.scene path/To/FileB.scene numberOfFrames sumErrorForAllObjects generateAngleObjects \n");
     return 0;
   }
    return compareTrajectoryFiles("comparison.txt",argv[1],argv[2],atoi(argv[3]),atoi(argv[4]),atoi(argv[5]));
}
