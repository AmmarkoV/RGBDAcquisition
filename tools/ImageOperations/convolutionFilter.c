#include "convolutionFilter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


static inline int isFilterDimensionOdd(unsigned int dimension)
{
 return ( dimension % 2 !=0 );
}


static inline void doFilterKernel3ch (
                             float * sourceKernelPosition , float * kernelStart , unsigned int kernelWidth , unsigned int kernelHeight , float divisor ,
                             unsigned int sourceWidth ,
                             float * output
                          )
{
  // ================================================================
  unsigned int lineOffset = ( sourceWidth*3 ) - (kernelWidth*3) ;
  float * imagePTR = sourceKernelPosition;
  float * imageLineEnd = sourceKernelPosition + (3*kernelWidth);
  float * imageEnd = sourceKernelPosition + (sourceWidth*3*kernelHeight);

  float * kernelFPTR = kernelStart;
  //float * kernelFPTREnd = kernelStart + (kernelHeight*kernelWidth*3);

  float resR=0,resG=0,resB=0;

  while (imagePTR < imageEnd)
  {
    while (imagePTR < imageLineEnd)
    {
     resR+= (*kernelFPTR) * (*imagePTR); ++imagePTR;
     resG+= (*kernelFPTR) * (*imagePTR); ++imagePTR;
     resB+= (*kernelFPTR) * (*imagePTR); ++imagePTR;
     ++kernelFPTR;
    }
   imagePTR+=lineOffset;
   imageLineEnd+=( sourceWidth*3 );
  }

  output[0] = (float) resR / divisor;
  output[1] = (float) resG / divisor;
  output[2] = (float) resB / divisor;

 return;
}



int convolutionFilter3ChF(
                          float * target,  unsigned int targetWidth , unsigned int targetHeight ,
                          float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                          float * convolutionMatrix , unsigned int kernelWidth , unsigned int kernelHeight , float divisor
                         )
{
  if (convolutionMatrix==0) { return 0; }

  float * sourcePTR = source;
  float * sourceLimit = source+(sourceWidth*sourceHeight*3) ;
  float * targetPTR = target;

  if (
       (!isFilterDimensionOdd(kernelWidth)) ||
       (!isFilterDimensionOdd(kernelHeight))
     )
  {
    return 0;
  }


  unsigned int x=0,y=0;
  while (sourcePTR < sourceLimit)
  {
   float * sourceScanlinePTR = sourcePTR;
   float * sourceScanlineEnd = sourcePTR+(sourceWidth*3);

   float * targetScanlinePTR = targetPTR;
   float * targetScanlineEnd = targetPTR+(targetWidth*3);

   if (x+kernelWidth>=sourceWidth)
   {

   } else
   if (y+kernelHeight>=sourceHeight)
   {
      //We are on the right side of our buffer
   } else
   {
   float outputRGB[3];
   //Get all the valid configurations of the scanline
   while (sourceScanlinePTR < sourceScanlineEnd)
    {
      doFilterKernel3ch ( sourceScanlinePTR  , convolutionMatrix , kernelWidth , kernelHeight , divisor , sourceWidth , outputRGB );

      float * outputR = targetScanlinePTR + (targetWidth*3) + 3;
      float * outputG = outputR+1;
      float * outputB = outputG+1;

      *outputR = outputRGB[0];
      *outputG = outputRGB[1];
      *outputB = outputRGB[2];

      ++x;
      sourceScanlinePTR+=3;
      targetScanlinePTR+=3;
    }
     sourcePTR = sourceScanlineEnd-3;
     targetPTR = targetScanlineEnd-3;

   }

   //Keep X,Y Relative positiions
   ++x;
   if (x>=sourceWidth) { x=0; ++y; }
   sourcePTR+=3;
   targetPTR+=3;
  }

 return 1;
}










/*
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -



     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
*/







static inline void doFilterKernel1ch (
                             float * sourceKernelPosition , float * kernelStart , unsigned int kernelWidth , unsigned int kernelHeight , float divisor ,
                             unsigned int sourceWidth ,
                             float * output
                          )
{
  // ================================================================
  unsigned int lineOffset = ( sourceWidth*1 ) - (kernelWidth*1) ;
  float * imagePTR = sourceKernelPosition;
  float * imageLineEnd = sourceKernelPosition + (1*kernelWidth);
  float * imageEnd = sourceKernelPosition + (sourceWidth*1*kernelHeight);

  float * kernelFPTR = kernelStart;
  //float * kernelFPTREnd = kernelStart + (kernelHeight*kernelWidth*1);

  float res=0;
  //divisor=1;
  while (imagePTR < imageEnd)
  {
    while (imagePTR < imageLineEnd)
    {
     res+= (*kernelFPTR) * (*imagePTR);
     ++imagePTR;
     ++kernelFPTR;
    }
   imagePTR+=lineOffset;
   imageLineEnd+=( sourceWidth*1 );
  }

  output[0] = (float) res / divisor;

 return;
}

int convolutionFilter1ChF(
                          float * target,  unsigned int targetWidth , unsigned int targetHeight ,
                          float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                          float * convolutionMatrix , unsigned int kernelWidth , unsigned int kernelHeight , float * divisor
                         )
{

  #define channels 1
  if (convolutionMatrix==0) { return 0; }
  float * sourcePTR = source;
  float * sourceLimit = source+(sourceWidth*sourceHeight*channels) ;
  float * targetPTR = target;
  float * targetLimit = target+(targetWidth*targetHeight*channels) ;

  unsigned int halfKernelWidth = (unsigned int) kernelWidth/2;
  unsigned int halfKernelHeight = (unsigned int) kernelHeight/2;
  unsigned int targetOffsetForAnchorOfKernel = ( targetWidth * channels * halfKernelHeight ) + ( halfKernelWidth * channels ) ;

  if (
       (!isFilterDimensionOdd(kernelWidth)) ||
       (!isFilterDimensionOdd(kernelHeight))
     ) { fprintf(stderr,"Dimensions are not odd (%u,%u) , not doing convolutionFilter1ChF \n",kernelWidth,kernelHeight); return 0; }

  if (*divisor == 0.0) { fprintf(stderr,"Kernel Divisor cannot be zero , not doing convolutionFilter1ChF \n"); return 0; }

  unsigned int x=0,y=0;
  while (sourcePTR < sourceLimit)
  {
   float * sourceScanlinePTR = sourcePTR  , *sourceScanlineEnd = sourcePTR+(sourceWidth*channels);
   float * targetScanlinePTR = targetPTR  , *targetScanlineEnd = targetPTR+(targetWidth*channels);

   if (x+kernelWidth>=sourceWidth)    {  } else
   if (y+kernelHeight>=sourceHeight)  {  } else
   {
    float outputRGB[channels];
    //Get all the valid configurations of the scanline
    while (sourceScanlinePTR < sourceScanlineEnd)
    {
      doFilterKernel1ch ( sourceScanlinePTR  , convolutionMatrix , kernelWidth , kernelHeight , *divisor , sourceWidth , outputRGB );

      float * output = targetScanlinePTR + targetOffsetForAnchorOfKernel;
      *output = *outputRGB;


      ++x;
      sourceScanlinePTR+=channels;
      targetScanlinePTR+=channels;
    }
     sourcePTR = sourceScanlineEnd-channels;
     targetPTR = targetScanlineEnd-channels;

   }

   //Keep X,Y Relative positiions
   ++x;
   if (x>=sourceWidth) { x=0; ++y; }
   sourcePTR+=channels;
   targetPTR+=channels;
  }

 return 1;
}


