#include "learnImage.h"
#include <stdio.h>
#include <stdlib.h>
#include "../Codecs/codecs.h"


struct pixel
{
  unsigned int * indexes;
  unsigned int * compressed;
  unsigned int numberOfPixelsPresent;
  unsigned int numberOfPixelsAbsent;
};



struct imageOccupationGrid
{
  struct pixel * pixels;
};









int learnImage(const char * filename ,unsigned int numberOfHorizontalTiles,unsigned int numberOfVerticalTiles)
{
   struct Image * inputImage = readImage(filename,guessFilenameTypeStupid(filename),0);
   if (inputImage!=0)
   {
     unsigned int tileWidth =inputImage->width/numberOfHorizontalTiles;
     unsigned int tileHeight=inputImage->height/numberOfVerticalTiles;
     fprintf(stderr,"Sucessfully opened image ( %ux%u ) , each tile is %ux%u ..\n",inputImage->width,inputImage->height,tileWidth,tileHeight);


     unsigned int x,y,i=0;
     for (y=0; y<numberOfVerticalTiles; y++)
     {
      for (x=0; x<numberOfHorizontalTiles; x++)
      {
        struct Image * part = createImageBitBlt(inputImage,x*tileWidth , y*tileHeight , tileWidth, tileHeight);
        if (part!=0)
        {
          char tileString[512]={0};
          snprintf(tileString,512,"tile-%05d.jpg",i);
          writeImageFile(part,JPG_CODEC,tileString);
          destroyImage(part);
        }
       ++i;
      }
     }


     destroyImage(inputImage);
     return 1;
   }
  return 0;
}
