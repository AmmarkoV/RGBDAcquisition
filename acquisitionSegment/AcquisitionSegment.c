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
     x=0;
      while (sourcePixels<sourcePixelsLineEnd)
      {
        R = sourcePixels++;
        G = sourcePixels++;
        B = sourcePixels++;

       if  (
             (segConf->minR <= *R) && (*R <= segConf->maxR)  &&
              (segConf->minG <= *G) && (*G <= segConf->maxG)  &&
               (segConf->minB <= *B) && (*B <= segConf->maxB) &&

               (segConf->minX <= x) && ( x<= segConf->maxX)
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

short * segmentDepthFrame(short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf)
{
 short * target = (short *) malloc( width * height * sizeof(short));
 if ( target == 0) { return 0; }
 memset(target,0,width*height*sizeof(short));

 unsigned int sourceWidthStep = width;
 unsigned int targetWidthStep = width;
 unsigned int posX = segConf->minX;
 unsigned int posY = segConf->minY;
 width = segConf->maxX-segConf->minX;
 height = segConf->maxY-segConf->minY;

 short * sourcePixelsStart   = (short*) source + ( (posX) + posY * sourceWidthStep );
 short * sourcePixelsLineEnd = sourcePixelsStart + (width);
 short * sourcePixelsEnd     = sourcePixelsLineEnd + ((height-1) * sourceWidthStep );
 short * sourcePixels = sourcePixelsStart;

 short * targetPixelsStart   = (short*) target + ( (posX) + posY * targetWidthStep );
 short * targetPixelsLineEnd = targetPixelsStart + (width);
 short * targetPixelsEnd     = targetPixelsLineEnd + ((height-1) * targetWidthStep );
 short * targetPixels = targetPixelsStart;

 short * depth;
 while (sourcePixels<sourcePixelsEnd)
 {
   while (sourcePixels<sourcePixelsLineEnd)
    {
     depth = sourcePixels++;

     if  ( (segConf->minDepth <= *depth) && (*depth <= segConf->maxDepth) )
       {
         *targetPixels=*depth; targetPixels++;
       } else
       {
         targetPixels++;
       }

     }
   sourcePixelsLineEnd+=sourceWidthStep;
   targetPixelsLineEnd+=targetWidthStep;
 }

 return target;
}
