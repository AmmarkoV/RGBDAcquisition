#include <stdio.h>
#include <stdlib.h>

#include "depthSelector.h"

#include "combineRGBAndDepthOutput.h"

#include "imageProcessing.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrix4x4Tools.h"
#include "../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/AmMatrix/matrixCalculations.h"


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

 float fx = 537.479600 , fy = 536.572920 , cx = 317.389787 ,cy = 236.118093;
 if ( calib->intrinsicParametersSet )
 {
   fx = calib->intrinsic[CALIB_INTR_FX];
   fy = calib->intrinsic[CALIB_INTR_FY];
   cx = calib->intrinsic[CALIB_INTR_CX];
   cy = calib->intrinsic[CALIB_INTR_CY];

   if (fx==0) { fx=1;}
   if (fy==0) { fy=1;}
 } else {fprintf(stderr,"No intrinsic parameters provided , bounding box segmentation will use default intrinsic values ( you probably dont want this )\n"); }

 double * m = alloc4x4Matrix();
 if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform bounding box selection\n"); } else
 {
  create4x4IdentityMatrix(m);
  if ( calib->extrinsicParametersSet )
     {
        convertRodriguezAndTranslationToOpenGL4x4DMatrix(m, calib->extrinsicRotationRodriguez , calib->extrinsicTranslation);
        fprintf(stderr,"Is this correct , ? shouldnt the matrix be the other way around ? \n");
        transpose4x4MatrixD(m);
     }
  else {fprintf(stderr,"No extrinsic parameters provided , bounding box segmentation will use default coordinate system \n"); }

  double raw3D[4]={0};
  double world3D[4]={0};


  sourcePixelsStart   = (unsigned short*) sourceCopy + ( (posX) + posY * sourceWidthStep );
  sourcePixelsLineEnd = sourcePixelsStart + (width);
  sourcePixelsEnd     = sourcePixelsLineEnd + ((height-1) * sourceWidthStep );
  sourcePixels = sourcePixelsStart;

  selectedPtr   = selectedDepth;
  sourcePixels = sourcePixelsStart;
  sourcePixelsLineEnd = sourcePixelsStart + (width);
  x=0; y=0;
  depth=0;
  while (sourcePixels<sourcePixelsEnd)
  {
   while (sourcePixels<sourcePixelsLineEnd)
    {
     depth = sourcePixels++;

     if (  (*selectedPtr!=0)  )  //  &&  (*depth != 0)
     {

      raw3D[0] = (double) (x - cx) * (*depth) / fx;
      raw3D[1] = (double) (y - cy) * (*depth) / fy;
      raw3D[2] = (double) *depth;
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
 float fx = 537.479600 , fy = 536.572920 , cx = 317.389787 ,cy = 236.118093;
 if ( calib->intrinsicParametersSet )
 {
   fx = calib->intrinsic[CALIB_INTR_FX];
   fy = calib->intrinsic[CALIB_INTR_FY];
   cx = calib->intrinsic[CALIB_INTR_CX];
   cy = calib->intrinsic[CALIB_INTR_CY];

   if (fx==0) { fx=1;}
   if (fy==0) { fy=1;}
 } else {fprintf(stderr,"No intrinsic parameters provided , bounding box segmentation will use default intrinsic values ( you probably dont want this )\n"); }

 double * m = alloc4x4Matrix();
 if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform bounding box selection\n"); } else
 {
  create4x4IdentityMatrix(m);
  if ( calib->extrinsicParametersSet )
       {
        convertRodriguezAndTranslationToOpenGL4x4DMatrix(m, calib->extrinsicRotationRodriguez , calib->extrinsicTranslation);
        fprintf(stderr,"Is this correct , ? shouldnt the matrix be the other way around ? \n");
        transpose4x4MatrixD(m);
       }
  else {fprintf(stderr,"No extrinsic parameters provided , bounding box segmentation will use default coordinate system \n"); }

  double raw3D[4]={0};
  double world3D[4]={0};


    float p1[3]; p1[0]=(float) segConf->p1[0]; p1[1]=(float) segConf->p1[1]; p1[2]=(float) segConf->p1[2];
    float p2[3]; p2[0]=(float) segConf->p2[0]; p2[1]=(float) segConf->p2[1]; p2[2]=(float) segConf->p2[2];
    float p3[3]; p3[0]=(float) segConf->p3[0]; p3[1]=(float) segConf->p3[1]; p3[2]=(float) segConf->p3[2];
    float pN[3]={  p2[0] , p2[1]+5 , p2[2] };
    float normal[3]={0.0 , 0.0 , 0.0 };

    crossProduct( p1 , p2  , p3  , normal);


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
  while (sourcePixels<sourcePixelsEnd)
  {
   while (sourcePixels<sourcePixelsLineEnd)
    {
     depth = sourcePixels++;

     if (  (*selectedPtr!=0)  )  //  &&  (*depth != 0)
     {
      raw3D[0] = (double) (x - cx) * (*depth) / fx;
      raw3D[1] = (double) (y - cy) * (*depth) / fy;
      raw3D[2] = (double) *depth;
      raw3D[3] = (double) 1.0;

      transform3DPointUsing4x4Matrix(world3D,m,raw3D);

      pN[0]=(float) world3D[0];
      pN[1]=(float) world3D[1];
      pN[2]=(float) world3D[2];
      float result = signedDistanceFromPlane(p2, normal , pN);

      if (  result<=0.0 )  { *selectedPtr=0; } //Denied

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
