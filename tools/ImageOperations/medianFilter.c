#include "medianFilter.h"






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
     sum=kernelPTR;    ++kernelPTR;
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

  for (c=0; c<numberOfElements; c++)
    { labelBuffer[c]=c; }



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







  output[0]=sortingBuffer[elementToPick];
  output[1]=*kernelStart; ++kernelStart;
  output[2]=*kernelStart;
}


int medianFilter(
                 unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                 unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                 unsigned int blockWidth , unsigned int blockHeight
                )
{
  unsigned char * sourcePTR = source;
  unsigned char * sourceLimit = source+(sourceWidth*sourceHeight*3) ;
  unsigned char * targetPTR = target;
  unsigned int x=0,y=0;


  unsigned int elementToPick = (blockWidth * blockHeight) / 2;
  unsigned char * sortingBuffer = (unsigned char *) malloc(sizeof(unsigned char)  * blockWidth * blockHeight );
  unsigned char * labelBuffer = (unsigned char *) malloc(sizeof(unsigned char)  * blockWidth * blockHeight );


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
