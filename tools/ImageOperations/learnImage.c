#include "learnImage.h"
#include <stdio.h>
#include <stdlib.h>
#include "../Codecs/codecs.h"

int learnImage(const char * filename , unsigned int tileWidth , unsigned int tileHeight)
{
   struct Image * inputImage = readImage(filename,guessFilenameTypeStupid(filename),0);
   if (inputImage!=0)
   {
     fprintf(stderr,"Sucessfully opened image ( %ux%u ) , each tile is %ux%u ..\n",inputImage->width,inputImage->height,tileWidth,tileHeight);


     destroyImage(inputImage);
     return 1;
   }
  return 0;
}
