#include "convolutionFilter.h"


inline int isFilterDimensionOdd(unsigned int dimension)
{
 return ( dimension % 2 !=0 );
}


inline void doFilterKernel (
                             float * sourceKernelPosition , float * kernelStart , unsigned int kernelWidth , unsigned int kernelHeight , float divisor ,
                             unsigned int sourceWidth ,
                             unsigned char * output
                          )
{
  // ================================================================
  unsigned int lineOffset = ( sourceWidth*3 ) - (kernelWidth*3) ;
  float * imagePTR = sourceKernelPosition;
  float * imageLineEnd = sourceKernelPosition + (3*kernelWidth);
  float * imageEnd = sourceKernelPosition + (sourceWidth*3*kernelHeight);

  float * kernelFPTR = kernelStart;
  float * kernelFPTREnd = kernelStart + (kernelHeight*kernelWidth*3);

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
      doFilterKernel ( sourceScanlinePTR  , convolutionMatrix , kernelWidth , kernelHeight , divisor , sourceWidth , outputRGB );

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



int convolutionFilter1ChF(
                          float * target,  unsigned int targetWidth , unsigned int targetHeight ,
                          float * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                          float * convolutionMatrix , unsigned int kernelWidth , unsigned int kernelHeight , float divisor
                         )
{
  return 0;
}
