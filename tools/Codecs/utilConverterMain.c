#include <stdio.h>
#include <stdlib.h>

#include "codecs.h"

int main(int argc, char *argv[])
{
    if (argc<3)
    {
      fprintf(stderr,"Not enough arguments");
      return 1;
    } else
    if (argc>3)
    {
      fprintf(stderr,"Too many arguments");
      return 1;
    }

    printf("Converting %s to %s !\n",argv[2],argv[3]);
      convertCodecImages(argv[2],argv[3]);
    return 0;
}
