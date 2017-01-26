#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "colorSelector.h"

#include "combineRGBAndDepthOutput.h"

#include "imageProcessing.h"
#include "../tools/AmMatrix/matrix4x4Tools.h"
#include "../tools/AmMatrix/matrixCalculations.h"


int keepFirstRGBFrame(unsigned char * source ,  unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf)
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
  fprintf(stderr,"Dropped %lu of %u \n",dropped,width*height);

  return 1;
}


int removeFloodFillBeforeProcessing(unsigned char * source , unsigned int width , unsigned int height , struct SegmentationFeaturesRGB * segConf  )
{
  if (segConf->floodErase.totalPoints==0) { return 0; }

  unsigned char * target = (unsigned char *) malloc( width * height * 3 * sizeof(unsigned char));
  if ( target == 0) {  return 0; }
  memset(target,0,width*height*3*sizeof(unsigned char));

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

 free(target);

  return 1;
}

int justSelectAllRGBPixels(struct SegmentationFeaturesRGB * segConf , unsigned int width , unsigned int height)
{
  if (
       (segConf->floodErase.totalPoints==0) &&
       (segConf->enableRGBMotionDetection==0) &&
       (segConf->minR == 0) && ( 255 <= segConf->maxR) &&
       (segConf->minG == 0) && ( 255 <= segConf->maxG) &&
       (segConf->minB == 0) && ( 255 <= segConf->maxB) &&
       (segConf->enableReplacingColors == 0 ) &&
       (segConf->minX==0) && (segConf->maxX >= width) &&
       (segConf->minY==0) && (segConf->maxY >= height)
     )
  {

      return 1;
  }

  return 0;
}

unsigned char * selectSegmentationForRGBFrame(unsigned char * source , unsigned int inputFrameWidth , unsigned int inputFrameHeight , struct SegmentationFeaturesRGB * segConf, struct calibration * calib,unsigned int * selectedPixels)
{
 unsigned int width = inputFrameWidth;
 unsigned int height = inputFrameHeight;

 //This will be our response segmentation
 unsigned char * selectedRGB   = (unsigned char*) malloc(width*height*sizeof(unsigned char));
 if (selectedRGB==0) { fprintf(stderr,"Could not allocate memory for RGB Selection\n"); return 0; }

 //if we don't need to segment , conserve our CPU
 if (justSelectAllRGBPixels(segConf,width,height))
    {
      fprintf(stderr,"======== Just Selecting All RGB Frame ======== \n");
      *selectedPixels=width*height;
      memset(selectedRGB,1,width*height*sizeof(unsigned char));
      return selectedRGB;
    }

 //We initially disqualify ( unselect ) the whole image , so we only PICK things that are ok
 *selectedPixels=0;   memset(selectedRGB,0,width*height*sizeof(unsigned char));

 //In case our bounds are impossible we get an unselected image..!
 if ( segConf->maxX > width )  { segConf->maxX = width; }
 if ( segConf->maxY > height ) { segConf->maxY = height; }
 if ( segConf->minX > segConf->maxX ) { return selectedRGB; }
 if ( segConf->minY > segConf->maxY ) { return selectedRGB; }


 unsigned char * sourceCopy = (unsigned char *) malloc( width * height * 3 * sizeof( unsigned char));
 if ( sourceCopy == 0) { fprintf(stderr,"Could not allocate a buffer to copy the source..\n"); return 0; }
 memcpy(sourceCopy,source,width*height*3*sizeof(char));

 removeFloodFillBeforeProcessing(sourceCopy,width,height,segConf);


 unsigned int posX = segConf->minX , posY = segConf->minY;
 unsigned int limX = segConf->maxX , limY = segConf->maxY;
 unsigned int patchWidth = limX-posX-1 , patchHeight = limY-posY-1;
 unsigned int sourceWidthStep = width * 3;

 unsigned char * sourcePixelsStart   = (unsigned char*) sourceCopy + ( (posX*3) + posY * sourceWidthStep );
 unsigned char * sourcePixelsLineEnd = sourcePixelsStart + ((patchWidth)*3);
 unsigned char * sourcePixelsEnd     = sourcePixelsLineEnd + ((patchHeight-1) * sourceWidthStep );
 unsigned char * sourcePixels = sourcePixelsStart;

 unsigned char * selectedPtrStart = selectedRGB + ( posX + (posY * width) ) ;
 unsigned char * selectedPtr = selectedPtrStart;


 register unsigned char selected;
 register unsigned char * R , * G , * B;
 while (sourcePixels<sourcePixelsEnd)
 {
   while (sourcePixels<sourcePixelsLineEnd)
      {
        R = sourcePixels++;
        G = sourcePixels++;
        B = sourcePixels++;

        selected=(
                       ( (*R!=segConf->replaceR) || (*G!=segConf->replaceG) || (*B!=segConf->replaceB) ) &&
                         (
                              (segConf->minR <= *R) && (*R <= segConf->maxR) &&
                              (segConf->minG <= *G) && (*G <= segConf->maxG) &&
                              (segConf->minB <= *B) && (*B <= segConf->maxB)
                         )
                  );

        *selectedPixels+=selected;
        *selectedPtr=selected;
        ++selectedPtr;
      }
   sourcePixelsStart+=sourceWidthStep;
   sourcePixels=sourcePixelsStart;
   sourcePixelsLineEnd+=sourceWidthStep;
   selectedPtrStart+=width;
   selectedPtr=selectedPtrStart;
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



  if (segConf->invert)
     { invertSelection(selectedRGB , inputFrameWidth ,   inputFrameHeight ,selectedPixels); }



 free(sourceCopy);
 return selectedRGB;
}

