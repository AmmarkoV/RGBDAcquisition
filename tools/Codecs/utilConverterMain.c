#include <stdio.h>
#include <stdlib.h>


#define USE_JPG_FILES 1
#define USE_PNG_FILES 1
#include "codecs.h"

int main(int argc, char *argv[])
{
    if (argc<3)
    {
      fprintf(stderr,"Not enough arguments\n");
      return 1;
    } else
    if (argc>3)
    {
      fprintf(stderr,"Too many arguments\n");
      return 1;
    }

    fprintf(stderr,"Converting %s to %s !\n",argv[1],argv[2]);
      convertCodecImages(argv[1],argv[2]);
    return 0;
}
