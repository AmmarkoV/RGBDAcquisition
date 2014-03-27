#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "depthSelector.h"

#include "combineRGBAndDepthOutput.h"

#include "imageProcessing.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrix4x4Tools.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrixCalculations.h"

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
  fprintf(stderr,"Dropped %u of %u \n",dropped,width*height);

  return 1;
}



int removeDepthFloodFillBeforeProcessing(unsigned short * source , unsigned short * target , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf  )
{
  if (segConf->floodErase.totalPoints==0) { return 0; }
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

  return 1;
}




unsigned char * selectSegmentationForDepthFrame(unsigned short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf , struct calibration * calib)
{
 unsigned short * sourceCopy = (unsigned short *) malloc( width * height * sizeof(unsigned short));
 if ( sourceCopy == 0) { return 0; }
 memcpy(sourceCopy,source,width*height*sizeof(unsigned short));


 unsigned short * target = (unsigned short *) malloc( width * height * sizeof(unsigned short));
 if ( target == 0) { free(sourceCopy); return 0; }
 memset(target,0,width*height*sizeof(short));

 removeDepthFloodFillBeforeProcessing(sourceCopy,target,width,height,segConf);

 if ( target != 0)  { free(target); target=0; }

 unsigned int sourceWidthStep = width;
 unsigned int targetWidthStep = width;
 unsigned int posX = segConf->minX;
 unsigned int posY = segConf->minY;
 width = segConf->maxX-segConf->minX;
 height = segConf->maxY-segConf->minY;

 unsigned short * sourcePixelsStart   = (unsigned short*) sourceCopy + ( (posX) + posY * sourceWidthStep );
 unsigned short * sourcePixelsLineEnd = sourcePixelsStart + (width);
 unsigned short * sourcePixelsEnd     = sourcePixelsLineEnd + ((height-1) * sourceWidthStep );
 unsigned short * sourcePixels = sourcePixelsStart;

 unsigned char * selectedDepth   = (unsigned char*) malloc(width*height*sizeof(unsigned char));
 if (selectedDepth==0) { fprintf(stderr,"Could not allocate memory for RGB Selection\n"); return 0; }
 memset(selectedDepth,0,width*height*sizeof(unsigned char));

 unsigned char * selectedPtr   = selectedDepth;

 unsigned int x =0;
 unsigned int y =0;






 unsigned short * depth=0;
 while (sourcePixels<sourcePixelsEnd)
 {
   while (sourcePixels<sourcePixelsLineEnd)
    {
     depth = sourcePixels++;

     if (*depth != 0)
     { //If there is a depth given for point
       if  ( (segConf->minDepth <= *depth) && (*depth <= segConf->maxDepth) ) { *selectedPtr=1; } else
                                                                              { *selectedPtr=0; }
     }

     ++selectedPtr;
     ++x;
     if (x>=width) { x=0; ++y;}
    }
   sourcePixelsLineEnd+=sourceWidthStep;
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
                 segConf->bboxX1,
                 segConf->bboxY1,
                 segConf->bboxZ1,
                 segConf->bboxX2,
                 segConf->bboxY2,
                 segConf->bboxZ2
         );

 double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(calib);
 if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform bounding box selection\n"); } else
 {
  double raw3D[4];
  double world3D[4];

  sourcePixelsStart   = (unsigned short*) sourceCopy + ( (posX) + posY * sourceWidthStep );
  sourcePixelsLineEnd = sourcePixelsStart + (width);
  sourcePixelsEnd     = sourcePixelsLineEnd + ((height-1) * sourceWidthStep );
  sourcePixels = sourcePixelsStart;

  selectedPtr   = selectedDepth;
  sourcePixels = sourcePixelsStart;
  sourcePixelsLineEnd = sourcePixelsStart + (width);
  x=0; y=0;
  depth=0;

  float x3D , y3D , z3D;

  while (sourcePixels<sourcePixelsEnd)
  {
   while (sourcePixels<sourcePixelsLineEnd)
    {
     depth = sourcePixels++;

     if (  (*selectedPtr!=0)  )  //  &&  (*depth != 0)
     {
       transform2DProjectedPointTo3DPoint(calib , x, y , *depth , &x3D , &y3D ,  &z3D);

       raw3D[0] = (double) x3D;
       raw3D[1] = (double) y3D;
       raw3D[2] = (double) z3D;
       raw3D[3] = (double) 1.0;

       transform3DPointUsing4x4Matrix(world3D,m,raw3D);

       if (
           (segConf->bboxX1<world3D[0])&& (segConf->bboxX2>world3D[0]) &&
           (segConf->bboxY1<world3D[1])&& (segConf->bboxY2>world3D[1]) &&
           (segConf->bboxZ1<world3D[2])&& (segConf->bboxZ2>world3D[2])
          )
     {   } else // If it was selected keep it selected
     { *selectedPtr=0; } //Denied
     }//If it was selected and not null project it into 3d Space

     ++selectedPtr;
     ++x;
     if (x>=width) { x=0; ++y;}
    }
   sourcePixelsLineEnd+=sourceWidthStep;
 }
  free4x4Matrix(&m); // This is the same as free(m); m=0;
 } //End of M allocated!

}
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
    float pN[3]={  p2[0] , p2[1]+5 , p2[2] };
    float normal[3]={0.0 , 0.0 , 0.0 };

    crossProductFrom3Points( p1 , p2  , p3  , normal);


    //fprintf(stderr,"signedDistanceFromPlane is %0.2f \n",signedDistanceFromPlane(segConf->p2, normal , pN));



  sourcePixelsStart   = (unsigned short*) sourceCopy + ( (posX) + posY * sourceWidthStep );
  sourcePixelsLineEnd = sourcePixelsStart + (width);
  sourcePixelsEnd     = sourcePixelsLineEnd + ((height-1) * sourceWidthStep );
  sourcePixels = sourcePixelsStart;

  selectedPtr   = selectedDepth;
  sourcePixels = sourcePixelsStart;
  sourcePixelsLineEnd = sourcePixelsStart + (width);
  x=0; y=0;
  depth=0;

  float x3D , y3D , z3D;

  while (sourcePixels<sourcePixelsEnd)
  {
   while (sourcePixels<sourcePixelsLineEnd)
    {
     depth = sourcePixels++;

     if (  (*selectedPtr!=0)  )  //  &&  (*depth != 0)
     {
      transform2DProjectedPointTo3DPoint(calib , x, y , *depth , &x3D , &y3D ,  &z3D);
      raw3D[0] = (double) x3D; raw3D[1] = (double) y3D; raw3D[2] = (double) z3D; raw3D[3] = (double) 1.0;

      transform3DPointUsing4x4Matrix(world3D,m,raw3D);

      pN[0]=(float) world3D[0];
      pN[1]=(float) world3D[1];
      pN[2]=(float) world3D[2];
      float result = signedDistanceFromPlane(p2, normal , pN);

      if (  result <= 0.0 + segConf->planeNormalOffset )  { *selectedPtr=0; } //Denied

     }//If it was selected and not null project it into 3d Space

     ++selectedPtr;
     ++x;
     if (x>=width) { x=0; ++y;}
    }
   sourcePixelsLineEnd+=sourceWidthStep;
 }
  free4x4Matrix(&m); // This is the same as free(m); m=0;
 } //End of M allocated!

 }
 //-----------------------------------------------------------------------------


 free(sourceCopy);
 return selectedDepth;
}
