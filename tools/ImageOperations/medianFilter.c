#include "medianFilter.h"

#include <stdio.h>
#include <stdlib.h>

struct sortingValues
{
 unsigned char grayscale;
 unsigned char * rPtr;
};



static inline void doMedianFilterKernel (
                                    unsigned char * kernelStart,
                                    unsigned int kernelWidth ,
                                    unsigned int kernelHeight ,
                                    unsigned int elementToPick ,

                                    unsigned int sourceWidth ,
                                    unsigned int sourceHeight ,

                                    struct sortingValues * sortingBuffer,

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

  struct sortingValues * sortingBufferPTR = sortingBuffer;
  unsigned int  sum=0;
  //Get all input in tmpBuf so that we only access it..
  while (kernelPTR < kernelEnd)
  {
    while (kernelPTR < kernelLineEnd)
    {
     sortingBufferPTR->rPtr = kernelPTR;
     sum=*kernelPTR;     ++kernelPTR;
     sum+=*kernelPTR;    ++kernelPTR;
     sum+=*kernelPTR;    ++kernelPTR;
     sortingBufferPTR->grayscale = (unsigned char) sum/3;
     ++sortingBufferPTR;
    }
   kernelPTR+=lineOffset;
   kernelLineEnd+=( sourceWidth*3 );
  }

  struct sortingValues t;
  int c, d;
  int numberOfElements = kernelWidth*kernelHeight;

  for (c = 1 ; c <= numberOfElements - 1; c++)
  {
    d = c;

    while ( d > 0 && sortingBuffer[d].grayscale < sortingBuffer[d-1].grayscale)
    {
      //Swap Values --------------------
      t          = sortingBuffer[d];
      sortingBuffer[d]   = sortingBuffer[d-1];
      sortingBuffer[d-1] = t;
      //---------------------------------

      d--;
    }
  }


  output[0]=*sortingBuffer[elementToPick].rPtr; ++sortingBuffer[elementToPick].rPtr;
  output[1]=*sortingBuffer[elementToPick].rPtr; ++sortingBuffer[elementToPick].rPtr;
  output[2]=*sortingBuffer[elementToPick].rPtr;
}


int medianFilter3ch(
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
  //unsigned char * sortingBuffer = (unsigned char *) malloc(sizeof(unsigned char)  * kernelWidth * kernelHeight );
  //unsigned char * labelBuffer = (unsigned char *) malloc(sizeof(unsigned char)  * kernelWidth * kernelHeight );

  struct sortingValues * sortingBuffer = (struct sortingValues *) malloc(sizeof(struct sortingValues)  * kernelWidth * kernelHeight );

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

 return 0;
}
