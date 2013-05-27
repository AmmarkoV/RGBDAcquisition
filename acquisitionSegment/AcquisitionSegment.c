#include <stdio.h>
#include <stdlib.h>
#include "AcquisitionSegment.h"


char * segmentRGBFrame(char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf)
{
 char * target = (char *) malloc( width * height * 3 * sizeof(char));
 if ( target == 0) { return 0; }
 memset(target,0,width*height*3*sizeof(char));

 unsigned int posX = 0;
 unsigned int posY = 0;
 unsigned int sourceWidthStep = width * 3;
 unsigned int targetWidthStep = width * 3;

 char * sourcePixelsStart   = (char*) source + ( (posX*3) + posY * sourceWidthStep );
 char * sourcePixelsLineEnd = sourcePixelsStart + (width*3);
 char * sourcePixelsEnd     = sourcePixelsLineEnd + ((height-1) * sourceWidthStep );
 char * sourcePixels = sourcePixelsStart;

 char * targetPixelsStart   = (char*) target + ( (posX*3) + posY * targetWidthStep );
 char * targetPixelsLineEnd = targetPixelsStart + (width*3);
 char * targetPixelsEnd     = targetPixelsLineEnd + ((height-1) * targetWidthStep );
 char * targetPixels = targetPixelsStart;

 unsigned int x=0 , y=0;
 char * R , * G , * B;
 while (sourcePixels<sourcePixelsEnd)
 {

   if ( (y<segConf->minY) || (y>segConf->maxY) )
   {
     sourcePixels+=sourceWidthStep;
     targetPixels+=targetWidthStep;
   } else
   {
      while (sourcePixels<sourcePixelsLineEnd)
      {
        R = sourcePixels++;
        G = sourcePixels++;
        B = sourcePixels++;

       if  (
             (segConf->minR <= *R) && (*R <= segConf->maxR)  &&
              (segConf->minG <= *G) && (*G <= segConf->maxG)  &&
               (segConf->minB <= *B) && (*B <= segConf->maxB)
           )
       {
         *targetPixels=*R; targetPixels++;
         *targetPixels=*G; targetPixels++;
         *targetPixels=*B; targetPixels++;
       } else
       {
         targetPixels+=3;
       }

        ++x;
      }
   }
   sourcePixelsLineEnd+=sourceWidthStep;
   targetPixelsLineEnd+=targetWidthStep;
   ++y;
 }



 return target;
}
