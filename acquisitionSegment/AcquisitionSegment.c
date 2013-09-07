#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "AcquisitionSegment.h"


int floodFill(unsigned char * target , unsigned int width , unsigned int height ,
                signed int pX , signed int pY , int threshold,
                unsigned char sR , unsigned char sG , unsigned char sB ,
                unsigned char R , unsigned char G , unsigned char B , int depth)
{
 if ( (pX<0) || (pY<0) || (pX>=width) || (pY>=height) ) { return 0; }
 if (depth>2000) { return 0; }

 if (target==0) { return 0; }
 if (width==0) { return 0; }
 if (height==0) { return 0; }

 unsigned char * source = (unsigned char *) target  + ( (pX*3) + pY * width*3 );

 unsigned char * tR = source; ++source;
 unsigned char * tG = source; ++source;
 unsigned char * tB = source;


  if ( ( *tR == R   ) &&  ( *tG == G   )  &&  ( *tB == B ) ) { return 0; }


  if (
       (( *tR > sR-threshold ) && ( *tR < sR+threshold )) &&
       (( *tG > sG-threshold ) && ( *tG < sG+threshold )) &&
       (( *tB > sB-threshold ) && ( *tB < sB+threshold ))
      )
      {
        *tR = R; *tG = G; *tB = B;

        floodFill(target,width,height, pX+1 , pY ,   threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX-1 , pY ,   threshold, sR , sG , sB , R , G , B ,depth+1);

        floodFill(target,width,height, pX , pY+1 ,   threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX , pY-1 ,   threshold, sR , sG , sB , R , G , B ,depth+1);

        floodFill(target,width,height, pX+1 , pY+1 , threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX-1 , pY-1 , threshold, sR , sG , sB , R , G , B ,depth+1);

        floodFill(target,width,height, pX-1 , pY+1 , threshold, sR , sG , sB , R , G , B ,depth+1);
        floodFill(target,width,height, pX+1 , pY-1 , threshold, sR , sG , sB , R , G , B ,depth+1);
      }

   return 1;
}


int removeFloodFillBeforeProcessing(unsigned char * source , unsigned char * target , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf  )
{
  if (segConf->floodErase.totalPoints==0) { return 0; }
  unsigned char sR , sG, sB ;

  int i=0;
  for (i=0; i<segConf->floodErase.totalPoints; i++)
  {
     if (segConf->floodErase.source)
     {

       unsigned char * srcColor = (unsigned char *) source  + ( (segConf->floodErase.pX[i]*3) + segConf->floodErase.pY[i] * width*3 );
       sR = *srcColor; ++srcColor;
       sG = *srcColor; ++srcColor;
       sB = *srcColor; ++srcColor;
       //fprintf(stderr,"Flood Filling Before %u  - %u,%u thresh(%u) \n",i,segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i]);
       //fprintf(stderr,"Src Color %u,%u,%u \n",sR,sG,sB);

       floodFill(source , width, height ,
                 segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i],
                 sR,sG,sB , 0 , 0 , 0    , 0 );
     }
     //fprintf(stderr,"Flood Filled Before %u  - %u,%u thresh(%u) \n",i,segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i]);
  }

  return 1;
}


int removeFloodFillAfterProcessing(unsigned char * source  , unsigned char * target  , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf  )
{
  if (segConf->floodErase.totalPoints==0) { return 0; }
  unsigned char sR , sG, sB ;

  int i=0;
  for (i=0; i<segConf->floodErase.totalPoints; i++)
  {
     //fprintf(stderr,"Flood Filling After %u  - %u,%u thresh(%u) \n",i,segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i]);
     if (segConf->floodErase.target)
     {

       unsigned char * srcColor = (unsigned char *) target  + ( (segConf->floodErase.pX[i]*3) + segConf->floodErase.pY[i] * width*3 );
       sR = *srcColor; ++srcColor;
       sG = *srcColor; ++srcColor;
       sB = *srcColor; ++srcColor;

       floodFill(target , width, height ,
                 segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i],
                 sR,sG,sB , 0 , 0 , 0    , 0);
     }

     //fprintf(stderr,"Flood Filled After %u  - %u,%u thresh(%u) \n",i,segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i]);
  }

  return 1;
}




char * segmentRGBFrame(char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf)
{
 unsigned char * sourceCopy = (unsigned char *) malloc( width * height * 3 * sizeof( unsigned char));
 if ( sourceCopy == 0) { return 0; }
 memcpy(sourceCopy,source,width*height*3*sizeof(char));


 char * target = (char *) malloc( width * height * 3 * sizeof(char));
 if ( target == 0) {  free(sourceCopy); return 0; }
 memset(target,0,width*height*3*sizeof(char));

 removeFloodFillBeforeProcessing(sourceCopy,target,width,height,segConf);


 unsigned int posX = 0;
 unsigned int posY = 0;
 unsigned int sourceWidthStep = width * 3;
 unsigned int targetWidthStep = width * 3;

 char * sourcePixelsStart   = (char*) sourceCopy + ( (posX*3) + posY * sourceWidthStep );
 char * sourcePixelsLineEnd = sourcePixelsStart + (width*3);
 char * sourcePixelsEnd     = sourcePixelsLineEnd + ((height-1) * sourceWidthStep );
 char * sourcePixels = sourcePixelsStart;

 char * targetPixelsStart   = (char*) target + ( (posX*3) + posY * targetWidthStep );
 char * targetPixelsLineEnd = targetPixelsStart + (width*3);
 char * targetPixelsEnd     = targetPixelsLineEnd + ((height-1) * targetWidthStep );
 char * targetPixels = targetPixelsStart;

 unsigned int x=0 , y=0;
 unsigned char * R , * G , * B;
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
         if (segConf->enableReplacingColors)
         {
           *targetPixels=segConf->replaceR; targetPixels++;
           *targetPixels=segConf->replaceG; targetPixels++;
           *targetPixels=segConf->replaceB; targetPixels++;
         }
          else
         {
          *targetPixels=*R; targetPixels++;
          *targetPixels=*G; targetPixels++;
          *targetPixels=*B; targetPixels++;
         }
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


 removeFloodFillAfterProcessing(sourceCopy , target,width,height,segConf  );

 free(sourceCopy);
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


//acquisitionGetDepth3DPointAtXY(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int x2d, unsigned int y2d , float *x, float *y , float *z  );

 unsigned int x =0;
 unsigned int y =0;

 float x3D;
 float y3D;
 float z3D;

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


       ++x;
       if (x>=width) { x=0; ++y;}
     }
   sourcePixelsLineEnd+=sourceWidthStep;
   targetPixelsLineEnd+=targetWidthStep;
 }

 return target;
}




int getDepthBlobAverage(float * centerX , float * centerY , float * centerZ , short * frame , unsigned int width , unsigned int height)
{
  unsigned int x=0,y=0;


  unsigned long sumX=0,sumY=0,sumZ=0,samples=0;

   short * sourcePixels   = (short*) frame ;
   short * sourcePixelsEnd   =  sourcePixels + width * height ;

   while (sourcePixels<sourcePixelsEnd)
   {
     if (*sourcePixels != 0)
     {
       sumX+=x;
       sumY+=y;
       sumZ+=*sourcePixels;
       ++samples;
     }
     ++sourcePixels;
     ++x;
     if (x==width) { ++y; x=0;}
   }

   *centerX = (float) sumX / samples;
   *centerY = (float) sumY / samples;
   *centerZ = (float) sumZ / samples;
   return 1;
}

