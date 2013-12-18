#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "colorSelector.h"

#include "combineRGBAndDepthOutput.h"

#include "imageProcessing.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrix4x4Tools.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrixCalculations.h"


int keepFirstRGBFrame(unsigned short * source ,  unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf)
{
  if (segConf->firstRGBFrame==0)
  {
      segConf->firstRGBFrameByteSize = width * height * 3 * sizeof (unsigned char);
      segConf->firstRGBFrame = (unsigned char * ) malloc( segConf->firstRGBFrameByteSize );

      if (segConf->firstRGBFrame !=0 )
      {
       memcpy(segConf->firstRGBFrame , source , segConf->firstRGBFrameByteSize);
       return 1;
      }
  }
  return 0;
}


int selectBasedOnRGBMovement(unsigned char  * selection,unsigned char * baseRGB , unsigned char * currentRGB ,
                             unsigned int thresholdR , unsigned int thresholdG, unsigned int thresholdB ,  unsigned int width , unsigned int height  )
{
  fprintf(stderr,"selectBasedOnRGBMovement is executed with a threshold of R%u G%u B%u , ( %u x %u ) \n",thresholdR , thresholdG, thresholdB,width,height);

  unsigned long dropped=0;
  unsigned char * baseRGBPTR  = baseRGB;
  unsigned char * currentRGBPTR  = currentRGB;
  unsigned char * selectionPTR  = selection;
  unsigned char * selectionLimit  = selection + width*height;
  unsigned char channelMoving=0;

  while (selectionPTR<selectionLimit)
  {

    channelMoving=0;
    if ( (*currentRGBPTR > *baseRGBPTR) && (*currentRGBPTR > *baseRGBPTR + thresholdR) ) { ++channelMoving;  } else
    if ( (*currentRGBPTR < *baseRGBPTR) && (*currentRGBPTR + thresholdR < *baseRGBPTR) ) { ++channelMoving;  }
    ++currentRGBPTR; ++baseRGBPTR;

    if ( (*currentRGBPTR > *baseRGBPTR) && (*currentRGBPTR > *baseRGBPTR + thresholdG) ) { ++channelMoving;  } else
    if ( (*currentRGBPTR < *baseRGBPTR) && (*currentRGBPTR + thresholdG < *baseRGBPTR) ) { ++channelMoving;  }
    ++currentRGBPTR; ++baseRGBPTR;

    if ( (*currentRGBPTR > *baseRGBPTR) && (*currentRGBPTR > *baseRGBPTR + thresholdB) ) { ++channelMoving;  } else
    if ( (*currentRGBPTR < *baseRGBPTR) && (*currentRGBPTR + thresholdB < *baseRGBPTR) ) { ++channelMoving;  }
    ++currentRGBPTR; ++baseRGBPTR;


    if (channelMoving<2)
       {
         *selectionPTR = 0;
         ++dropped;
       }

    ++selectionPTR;
  }
  fprintf(stderr,"Dropped %u of %u \n",dropped,width*height);

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

       if (
            !floodFill(source , width, height ,
                       segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i],
                       sR,sG,sB ,
                       segConf->replaceR , segConf->replaceG , segConf->replaceB
                       , 0 )
          )
          {
             fprintf(stderr,"Failed to flood fill , bad input..\n");
          }
     }
     //fprintf(stderr,"Flood Filled Before %u  - %u,%u thresh(%u) \n",i,segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i]);
  }

  return 1;
}





unsigned char * selectSegmentationForRGBFrame(unsigned char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf, struct calibration * calib)
{
 unsigned char * sourceCopy = (unsigned char *) malloc( width * height * 3 * sizeof( unsigned char));
 if ( sourceCopy == 0) { return 0; }
 memcpy(sourceCopy,source,width*height*3*sizeof(char));


 unsigned char * target = (unsigned char *) malloc( width * height * 3 * sizeof(unsigned char));
 if ( target == 0) {  free(sourceCopy); return 0; }
 memset(target,0,width*height*3*sizeof(char));

 removeFloodFillBeforeProcessing(sourceCopy,target,width,height,segConf);

 //TODO: REATTACH FLOOD FILL!
 unsigned int posX = 0;
 unsigned int posY = 0;
 unsigned int sourceWidthStep = width * 3;

 char * sourcePixelsStart   = (char*) sourceCopy + ( (posX*3) + posY * sourceWidthStep );
 char * sourcePixelsLineEnd = sourcePixelsStart + (width*3);
 char * sourcePixelsEnd     = sourcePixelsLineEnd + ((height-1) * sourceWidthStep );
 char * sourcePixels = sourcePixelsStart;

 unsigned char * selectedRGB   = (unsigned char*) malloc(width*height*sizeof(unsigned char));
 if (selectedRGB==0) { fprintf(stderr,"Could not allocate memory for RGB Selection\n"); return 0; }
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

       if (
            (*R==segConf->replaceR) &&
            (*G==segConf->replaceG) &&
            (*B==segConf->replaceB)
           ) { *selectedPtr=0; }
             else
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
   ++y;
 }



 if (segConf->enableRGBMotionDetection)
 {
  //In case we want motion detection we should record the first frame we have so that we can use it to select pixels
  if (! keepFirstRGBFrame(sourceCopy ,  width , height , segConf) )
  {
    selectBasedOnRGBMovement(selectedRGB, segConf->firstRGBFrame , sourceCopy ,
                             segConf->motionRThreshold , segConf->motionGThreshold , segConf->motionBThreshold ,  width , height  );
  }
 } else
 {
   if (segConf->firstRGBFrame!=0)
   {
     fprintf(stderr,"Freeing first frame for rgb motion detection\n");
     free(segConf->firstRGBFrame);
     segConf->firstRGBFrame=0;
   }
 }



 free(sourceCopy);
 return selectedRGB;
}

