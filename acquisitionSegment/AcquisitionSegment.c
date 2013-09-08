#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "combineRGBAndDepthOutput.h"


#include "AcquisitionSegment.h"

#include "imageProcessing.h"


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

/*
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
*/



unsigned char * selectSegmentationForRGBFrame(char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf)
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

 char * sourcePixelsStart   = (char*) sourceCopy + ( (posX*3) + posY * sourceWidthStep );
 char * sourcePixelsLineEnd = sourcePixelsStart + (width*3);
 char * sourcePixelsEnd     = sourcePixelsLineEnd + ((height-1) * sourceWidthStep );
 char * sourcePixels = sourcePixelsStart;

 unsigned char * selectedRGB   = (unsigned char*) malloc(width*height*sizeof(unsigned char));
 memset(selectedRGB,0,width*height*sizeof(unsigned char));

 unsigned char * selectedPtr   = selectedRGB;

 unsigned int x=0 , y=0;
 unsigned char * R , * G , * B;
 while (sourcePixels<sourcePixelsEnd)
 {

   if ( (y<segConf->minY) || (y>segConf->maxY) )
   {
     sourcePixels+=sourceWidthStep;
     selectedPtr+=width;
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
          *selectedPtr=1;
       } else
       {
          *selectedPtr=0;
       }

        ++selectedPtr;
        ++x;
      }
   }
   sourcePixelsLineEnd+=sourceWidthStep;
   selectedPtr+=width;
   ++y;
 }


 free(sourceCopy);
 return selectedRGB;
}



unsigned char * selectSegmentationForDepthFrame(short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf)
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


int   segmentRGBAndDepthFrame (    char * RGB ,
                                   unsigned short * Depth ,
                                   unsigned int width , unsigned int height ,
                                   struct SegmentationFeaturesRGB * segConfRGB ,
                                   struct SegmentationFeaturesDepth * segConfDepth,
                                   int combinationMode
                               )
{
  unsigned char * selectedRGB = selectSegmentationForRGBFrame(RGB , width , height , segConfRGB);
  unsigned char * selectedDepth = selectSegmentationForDepthFrame(Depth , width , height , segConfDepth);

  if ( combinationMode != DONT_COMBINE )
  {
     unsigned char *  combinedSelection = combineRGBAndDepthToOutput(selectedRGB,selectedDepth,combinationMode,width,height);
     executeSegmentationRGB(RGB,combinedSelection,width,height,segConfRGB);
     executeSegmentationDepth(Depth,combinedSelection,width,height,segConfDepth);
     if (combinedSelection!=0) { free(combinedSelection); combinedSelection=0; }
  } else
  {
     executeSegmentationRGB(RGB,selectedRGB,width,height,segConfRGB);
     executeSegmentationDepth(Depth,selectedDepth,width,height,segConfDepth);
  }


  if (selectedRGB!=0) { free(selectedRGB); selectedRGB=0; }
  if (selectedDepth!=0) { free(selectedDepth); selectedDepth=0; }

  return 1;
}
