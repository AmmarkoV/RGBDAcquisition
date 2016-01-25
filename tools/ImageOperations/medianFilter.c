#include "medianFilter.h"

#include <stdio.h>
#include <stdlib.h>





inline void doMedianFilterKernel (
                                    unsigned char * kernelStart,
                                    unsigned int kernelWidth ,
                                    unsigned int kernelHeight ,
                                    unsigned int elementToPick ,

                                    unsigned int sourceWidth ,
                                    unsigned int sourceHeight ,

                                    unsigned char * sortingBuffer,
                                    unsigned char * labelBuffer,

                                    unsigned char * output
                                  )
{
  output[0]=kernelStart[0];
  output[1]=kernelStart[1];
  output[2]=kernelStart[2];


  // ================================================================
  unsigned int lineOffset = ( sourceWidth*3 ) - (kernelWidth*3) ;
  unsigned char * kernelPTR = kernelStart;
  unsigned char * kernelLineEnd = kernelStart + (3*kernelWidth);
  unsigned char * kernelEnd = kernelStart + (sourceWidth*3*kernelHeight);

  unsigned char * sortingBufferPTR = sortingBuffer;
  unsigned int  sum=0;
  //Get all input in tmpBuf so that we only access it..
  while (kernelPTR < kernelEnd)
  {
    while (kernelPTR < kernelLineEnd)
    {
     sum=kernelPTR;     ++kernelPTR;
     sum+=kernelPTR;    ++kernelPTR;
     sum+=kernelPTR;    ++kernelPTR;
     *sortingBufferPTR = (unsigned char) sum/3;
     ++sortingBufferPTR;
    }
   kernelPTR+=lineOffset;
   kernelLineEnd+=( sourceWidth*3 );
  }


  int c, d, t;
  int numberOfElements = kernelWidth*kernelHeight;

  for (c=0; c<numberOfElements; c++) { labelBuffer[c]=c; }



  for (c = 1 ; c <= numberOfElements - 1; c++)
  {
    d = c;

    while ( d > 0 && sortingBuffer[d] < sortingBuffer[d-1])
    {
      //Swap Values --------------------
      t          = sortingBuffer[d];
      sortingBuffer[d]   = sortingBuffer[d-1];
      sortingBuffer[d-1] = t;
      //---------------------------------

      //Swap Buffers --------------------
      t          = labelBuffer[d];
      labelBuffer[d]   = labelBuffer[d-1];
      labelBuffer[d-1] = t;
      //---------------------------------

      d--;
    }
  }

  //This is wrong..
  unsigned int elementToPickLabel = labelBuffer[elementToPick];
  unsigned int x = elementToPickLabel / kernelHeight;
  unsigned int elementNegativeOffset = elementToPickLabel * kernelHeight;
  unsigned int y = 0;
  if ( numberOfElements>elementNegativeOffset )  { y =numberOfElements -elementNegativeOffset; }
  if ( x>=kernelWidth )  { x=kernelWidth; }
  if ( y>=kernelHeight ) { y=kernelHeight; }
  //fprintf(stderr,"x(%u/%u)  y(%u/%u) | " ,x,kernelWidth , y,kernelHeight);


  kernelPTR = kernelStart + ( y * sourceWidth * 3 ) + (x * 3);
  output[0]=*kernelPTR; ++kernelPTR;
  output[1]=*kernelPTR; ++kernelPTR;
  output[2]=*kernelPTR;
}


int medianFilter(
                 unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                 unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                 unsigned int kernelWidth , unsigned int kernelHeight
                )
{
  unsigned char * sourcePTR = source;
  unsigned char * sourceLimit = source+(sourceWidth*sourceHeight*3) ;
  unsigned char * targetPTR = target;
  unsigned int x=0,y=0;


  unsigned int elementToPick = (kernelWidth * kernelHeight) / 2;
  unsigned char * sortingBuffer = (unsigned char *) malloc(sizeof(unsigned char)  * kernelWidth * kernelHeight );
  unsigned char * labelBuffer = (unsigned char *) malloc(sizeof(unsigned char)  * kernelWidth * kernelHeight );


  while (sourcePTR < sourceLimit)
  {
   unsigned char * sourceScanlinePTR = sourcePTR;
   unsigned char * sourceScanlineEnd = sourcePTR+(sourceWidth*3);

   unsigned char * targetScanlinePTR = targetPTR;
   unsigned char * targetScanlineEnd = targetPTR+(targetWidth*3);

   if (x+kernelWidth>=sourceWidth)
   {

   } else
   if (y+kernelHeight>=sourceHeight)
   {
      //We are on the right side of our buffer
   } else
   {
   //Get all the valid configurations of the scanline
   while (sourceScanlinePTR < sourceScanlineEnd)
    {
      unsigned char outputRGB[3];

      doMedianFilterKernel (
                            sourceScanlinePTR,
                            kernelWidth ,
                            kernelHeight ,
                            elementToPick ,

                            sourceWidth ,
                            sourceHeight ,

                            sortingBuffer,
                            labelBuffer,

                            outputRGB
                           );

      unsigned char * outputR = targetScanlinePTR + (targetWidth*3) + 3;
      unsigned char * outputG = outputR+1;
      unsigned char * outputB = outputG+1;

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


 free(sortingBuffer);
 free(labelBuffer);

 return 0;
}
