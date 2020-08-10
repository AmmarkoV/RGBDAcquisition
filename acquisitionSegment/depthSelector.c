#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "depthSelector.h"

#include "combineRGBAndDepthOutput.h"

#include "imageProcessing.h"
#include "../tools/AmMatrix/matrix4x4Tools.h"
#include "../tools/AmMatrix/matrixCalculations.h"

int keepFirstDepthFrame(unsigned short * source ,  unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf)
{
  if (segConf->firstDepthFrame==0)
  {
      segConf->firstDepthFrameByteSize = width * height * sizeof (unsigned short);
      segConf->firstDepthFrame = (unsigned short * ) malloc( segConf->firstDepthFrameByteSize );

      if (segConf->firstDepthFrame !=0 )
      {
       memcpy(segConf->firstDepthFrame , source , segConf->firstDepthFrameByteSize);
       return 1;
      }
  }
  return 0;
}





int selectBasedOnMovement(unsigned char  * selection,unsigned short * baseDepth , unsigned short * currentDepth , unsigned int threshold ,  unsigned int width , unsigned int height  )
{
  fprintf(stderr,"selectBasedOnMovement is executed with a threshold of %u , ( %u x %u ) \n",threshold,width,height);

  unsigned long dropped=0;
  unsigned short * baseDepthPTR  = baseDepth;
  unsigned short * currentDepthPTR  = currentDepth;
  unsigned char * selectionPTR  = selection;
  unsigned char * selectionLimit  = selection + width*height;
  unsigned char pixelMoving=0;

  while (selectionPTR<selectionLimit)
  {

    pixelMoving=0;
    if (*currentDepthPTR > *baseDepthPTR)
    {
       if (*currentDepthPTR > *baseDepthPTR + threshold)
        {
         /*This voxel is the same as it was ( aka have not moved ) , so we select them out! */
         pixelMoving=1;
        }
    } else
    if (*currentDepthPTR < *baseDepthPTR)
    {
       if (*currentDepthPTR + threshold < *baseDepthPTR)
        {
         /*This voxel is the same as it was ( aka have not moved ) , so we select them out! */
         pixelMoving=1;
        }
    }

    if (!pixelMoving)
       {
         *selectionPTR = 0;
         ++dropped;
       }

    ++currentDepthPTR;
    ++baseDepthPTR;
    ++selectionPTR;
  }
  fprintf(stderr,"Dropped %lu of %u \n",dropped,width*height);

  return 1;
}



int removeDepthFloodFillBeforeProcessing(unsigned short * source  , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf  )
{
  if (segConf->floodErase.totalPoints==0) { return 0; }

  unsigned short * target = (unsigned short *) malloc( width * height * sizeof(unsigned short));
  if ( target == 0) {  return 0; }
  memset(target,0,width*height*sizeof(unsigned short));

  unsigned short sDepth ;

  int i=0;
  for (i=0; i<segConf->floodErase.totalPoints; i++)
  {
     if (segConf->floodErase.source)
     {

       unsigned short * srcDepth = (unsigned short *) source  + ( (segConf->floodErase.pX[i]) + segConf->floodErase.pY[i] * width );
       sDepth = *srcDepth; ++srcDepth;
       fprintf(stderr,"Flood Filling Before %u  - %u,%u thresh(%u) \n",i,segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i]);
       fprintf(stderr,"Src Depth is %u \n",sDepth);

       floodFillUShort(source , width, height ,
                       segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i],
                       sDepth, 0    , 0 );
     }
     //fprintf(stderr,"Flood Filled Before %u  - %u,%u thresh(%u) \n",i,segConf->floodErase.pX[i],segConf->floodErase.pY[i],segConf->floodErase.threshold[i]);
  }


 if ( target != 0)  { free(target); target=0; }

  return 1;
}



int justSelectAllDepthPixels(struct SegmentationFeaturesDepth * segConf , unsigned int width , unsigned int height)
{
  if (
       (segConf->enableBBox==0) &&
       (segConf->enablePlaneSegmentation==0) &&
       (segConf->enableDepthMotionDetection==0) &&
       (segConf->floodErase.totalPoints==0) &&
       (segConf->minDepth == 0) && ( 32000 <= segConf->maxDepth) &&
       (segConf->minX==0) && (segConf->maxX >= width) &&
       (segConf->minY==0) && (segConf->maxY >= height)
     )
  {
      return 1;
  }

  return 0;
}


unsigned char * selectSegmentationForDepthFrame(unsigned short * source , unsigned int inputFrameWidth , unsigned int inputFrameHeight , struct SegmentationFeaturesDepth * segConf , struct calibration * calib,unsigned int * selectedPixels)
{
 unsigned int width = inputFrameWidth;
 unsigned int height = inputFrameHeight;

 unsigned char * selectedDepth   = (unsigned char*) malloc(width*height*sizeof(unsigned char));
 if (selectedDepth==0) { fprintf(stderr,"Could not allocate memory for RGB Selection\n"); return 0; }

 //if we don't need to segment , conserve our CPU
 if (justSelectAllDepthPixels(segConf,width,height))
    {
      fprintf(stderr,"======== Just Selecting All Depth Frame ======== \n");
      *selectedPixels=width*height;
      memset(selectedDepth,1,width*height*sizeof(unsigned char));
      return selectedDepth;
    }

 //We initially disqualify ( unselect ) the whole image , so we only PICK things that are ok
 *selectedPixels=0; memset(selectedDepth,0,width*height*sizeof(unsigned char));


 unsigned short * sourceCopy = (unsigned short *) malloc( width * height * sizeof(unsigned short));
 if ( sourceCopy == 0) { fprintf(stderr,"Couldn ot allocate a buffer to copy depth frame for segmentation\n"); return 0; }
 memcpy(sourceCopy,source,width*height*sizeof(unsigned short));

 removeDepthFloodFillBeforeProcessing(sourceCopy,width,height,segConf);


 unsigned int sourceWidthStep = width;
 unsigned int posX = segConf->minX , posY = segConf->minY;
 unsigned int limX = segConf->maxX , limY = segConf->maxY;
 unsigned int patchWidth = limX-posX , patchHeight = limY-posY;
 width = segConf->maxX-segConf->minX;
 height = segConf->maxY-segConf->minY;

 unsigned short * sourcePixelsStart   = (unsigned short*) sourceCopy + ( (posX) + posY * sourceWidthStep );
 unsigned short * sourcePixelsLineEnd = sourcePixelsStart + (patchWidth);
 unsigned short * sourcePixelsEnd     = sourcePixelsLineEnd + ((patchHeight-1) * sourceWidthStep );
 unsigned short * sourcePixels = sourcePixelsStart;

 unsigned char * selectedPtrStart   = selectedDepth + ( (posX) + posY * sourceWidthStep );
 unsigned char * selectedPtr   = selectedPtrStart;

 unsigned int x =0 , y =0;

 register unsigned char selected;
 register unsigned short * depth=0;
 while (sourcePixels<sourcePixelsEnd)
 {
   while (sourcePixels<sourcePixelsLineEnd)
    {
     depth = sourcePixels++;
     selected = ((*depth != 0)&&((segConf->minDepth <= *depth) && (*depth <= segConf->maxDepth)));
     *selectedPtr=selected;
     *selectedPixels+=selected;

     ++selectedPtr;
    }
   sourcePixelsStart+=sourceWidthStep;
   sourcePixels=sourcePixelsStart;
   sourcePixelsLineEnd+=sourceWidthStep;
   selectedPtrStart+=sourceWidthStep;
   selectedPtr=selectedPtrStart;
 }


 if (segConf->enableDepthMotionDetection)
 {
  //In case we want motion detection we should record the first frame we have so that we can use it to select pixels
  if (! keepFirstDepthFrame(sourceCopy ,  width , height , segConf) )
  {
    selectBasedOnMovement(selectedDepth, segConf->firstDepthFrame , sourceCopy , segConf->motionDistanceThreshold  ,  width , height  );
  }
 } else
 {
   if (segConf->firstDepthFrame!=0)
   {
     fprintf(stderr,"Freeing first frame for depth motion detection\n");
     free(segConf->firstDepthFrame);
     segConf->firstDepthFrame=0;
   }
 }



// -------------------------------------------------------------------------------------------------
// --------------------------------- BOUNDING BOX SEGMENTATION -------------------------------------
// -------------------------------------------------------------------------------------------------


if (segConf->enableBBox)
{
  fprintf(stderr,"Selecting Bounding Box Min %0.2f %0.2f %0.2f -> Max %0.2f %0.2f %0.2f  \n",
                   segConf->bboxX1, segConf->bboxY1, segConf->bboxZ1,
                   segConf->bboxX2, segConf->bboxY2, segConf->bboxZ2 );

 double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(calib);
 if (m==0)
 {
   fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform bounding box selection\n");
 }
  else
 {
  sourcePixelsStart   = (unsigned short*) sourceCopy + ( (posX) + posY * sourceWidthStep );
  sourcePixelsLineEnd = sourcePixelsStart + (patchWidth);
  sourcePixelsEnd     = sourcePixelsLineEnd + ((patchHeight-1) * sourceWidthStep );
  sourcePixels = sourcePixelsStart;

  selectedPtrStart   = selectedDepth + ( (posX) + posY * sourceWidthStep );
  selectedPtr   = selectedPtrStart;

  double raw3D[4] ,  world3D[4];
  float x3D , y3D , z3D;
  x=posX; y=posY; depth=0;
  while (sourcePixels<sourcePixelsEnd)
  {
   while (sourcePixels<sourcePixelsLineEnd)
    {
     depth = sourcePixels++;

     if (*selectedPtr!=0)
     {
       transform2DProjectedPointTo3DPoint(calib , x, y , *depth , &x3D , &y3D ,  &z3D);

       raw3D[0] = (double) x3D;
       raw3D[1] = (double) y3D;
       raw3D[2] = (double) z3D;
       raw3D[3] = (double) 1.0;

       transform3DPointDVectorUsing4x4DMatrix(world3D,m,raw3D);

       selected=(
                   (segConf->bboxX1<world3D[0])&& (segConf->bboxX2>world3D[0]) &&
                   (segConf->bboxY1<world3D[1])&& (segConf->bboxY2>world3D[1]) &&
                   (segConf->bboxZ1<world3D[2])&& (segConf->bboxZ2>world3D[2])
                );

       if (!selected) { *selectedPtr=0; *selectedPixels-=1; } //New Pixel , just got denied
     }//If it was selected and not null project it into 3d Space

     ++selectedPtr;
     ++x;
    } // End Of Scanline Loop
   ++y; x=posX;
   sourcePixelsStart+=sourceWidthStep;
   sourcePixels=sourcePixelsStart;
   sourcePixelsLineEnd+=sourceWidthStep;
   selectedPtrStart+=sourceWidthStep;
   selectedPtr=selectedPtrStart;
 } //End of Master While loop
  free4x4DMatrix(&m); // This is the same as free(m); m=0;
 } //End of M allocated!
} // End of Bounding Box Segmentation
// -------------------------------------------------------------------------------------------------
// --------------------------------- BOUNDING BOX SEGMENTATION -------------------------------------
// -------------------------------------------------------------------------------------------------



if ( segConf->enablePlaneSegmentation )
 {
  double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(calib);
  if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform plane segmentation\n"); } else
  {
    double raw3D[4]={0};  raw3D[3] = (double) 1.0;
    double world3D[4]={0};

    float p1[3]; p1[0]=(float) segConf->p1[0]; p1[1]=(float) segConf->p1[1]; p1[2]=(float) segConf->p1[2];
    float p2[3]; p2[0]=(float) segConf->p2[0]; p2[1]=(float) segConf->p2[1]; p2[2]=(float) segConf->p2[2];
    float p3[3]; p3[0]=(float) segConf->p3[0]; p3[1]=(float) segConf->p3[1]; p3[2]=(float) segConf->p3[2];

    float pN[3]={ 0 }; //This is the buffer for each point tested
    float normal[3]={0.0 , 0.0 , 0.0 };

    if (segConf->doNotGenerateNormalFrom3Points)
    { //We have our normals ready
      p2[0]=segConf->center[0]; p2[1]=segConf->center[1];  p2[2]=segConf->center[2];
      normal[0]=segConf->normal[0]; normal[1]=segConf->normal[1]; normal[2]=segConf->normal[2];
    } else
    {
      crossProductFrom3Points( p1 , p2  , p3  , normal);
      segConf->center[0]=p2[0]; segConf->center[1]=p2[1]; segConf->center[2]=p2[2];
      segConf->normal[0]=normal[0]; segConf->normal[1]=normal[1]; segConf->normal[2]=normal[2];
    }

    fprintf(stderr,"Normal segmentation using point %f,%f,%f and normal %f,%f,%f\n",pN[0],pN[1],pN[2],normal[0],normal[1],normal[2]);


    //fprintf(stderr,"signedDistanceFromPlane is %0.2f \n",signedDistanceFromPlane(segConf->p2, normal , pN));
   sourcePixelsStart   = (unsigned short*) sourceCopy + ( (posX) + posY * sourceWidthStep );
   sourcePixelsLineEnd = sourcePixelsStart + (patchWidth);
   sourcePixelsEnd     = sourcePixelsLineEnd + ((patchHeight-1) * sourceWidthStep );
   sourcePixels = sourcePixelsStart;

   selectedPtrStart   = selectedDepth + ( (posX) + posY * sourceWidthStep );
   selectedPtr   = selectedPtrStart;

  float x3D , y3D , z3D , distanceFromPlane;
  x=posX; y=posY; depth=0;
  while (sourcePixels<sourcePixelsEnd)
  {
   while (sourcePixels<sourcePixelsLineEnd)
    {
     depth = sourcePixels++;

     if (*selectedPtr!=0)
     {
      transform2DProjectedPointTo3DPoint(calib , x, y , *depth , &x3D , &y3D ,  &z3D);
      raw3D[0] = (double) x3D; raw3D[1] = (double) y3D; raw3D[2] = (double) z3D; raw3D[3] = (double) 1.0;

      transform3DPointDVectorUsing4x4DMatrix(world3D,m,raw3D);

      pN[0]=(float) world3D[0];
      pN[1]=(float) world3D[1];
      pN[2]=(float) world3D[2];

      distanceFromPlane = signedDistanceFromPlane(p2, normal , pN);

      if (segConf->planeNormalSize!=0)
      {
         //We also have a ceiling defined
         if (  distanceFromPlane >= 0.0 + segConf->planeNormalOffset + segConf->planeNormalSize )  { *selectedPtr=0; } //Denied
      }
      if (  distanceFromPlane <= 0.0 + segConf->planeNormalOffset )  { *selectedPtr=0; } //Denied

     }//If it was selected and not null project it into 3d Space

     ++selectedPtr;
     ++x;
    } // End Of Scanline Loop
   ++y; x=posX;
   sourcePixelsStart+=sourceWidthStep;
   sourcePixels=sourcePixelsStart;
   sourcePixelsLineEnd+=sourceWidthStep;
   selectedPtrStart+=sourceWidthStep;
   selectedPtr=selectedPtrStart;
 } //End of Master While loop
  free4x4DMatrix(&m); // This is the same as free(m); m=0;
 } //End of M allocated!

 }
 //-----------------------------------------------------------------------------

  if (segConf->invert)
     { invertSelection(selectedDepth , inputFrameWidth ,   inputFrameHeight ,selectedPixels); }


 free(sourceCopy);
 return selectedDepth;
}
