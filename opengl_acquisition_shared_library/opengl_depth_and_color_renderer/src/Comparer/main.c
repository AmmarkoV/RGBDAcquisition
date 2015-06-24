#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../OGLRendererSandbox.h"


int main(int argc, char **argv)
{

  int i=0;
  for (i=0; i<argc; i++)
  {
     if (strcmp(argv[i],"-from1")==0) { } else
     if (strcmp(argv[i],"-from2")==0) {  }

  }

   compareTrajectoryFiles("comparison.txt",argv[1],argv[2]);

    return 0;
}
