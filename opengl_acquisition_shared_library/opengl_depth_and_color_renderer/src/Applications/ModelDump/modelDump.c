#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "../../Library/OGLRendererSandbox.h"


int main(int argc, char **argv)
{
   if (argc<3)
   {
     fprintf(stderr,"usage : ModelDump path/To/FileA.obj path/To/FileB.c \n");
     return 0;
   }

   unsigned int i=0;
   for (i=0; i<argc; argc++)
   {
     if (strcmp(argv[i],"")==0)
     {
         //TODO Dump TRI Bone Structure etc
     }
   }

    return dumpModelFile(argv[1],argv[2]);
}
