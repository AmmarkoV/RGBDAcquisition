#include "convolutionFilter.h"





int convolutionFilter(unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                      unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,

                      unsigned char * convolutionMatrix , unsigned int convolutionMatrixWidth , unsigned int convolutionMatrixHeight , unsigned int divisor ,

                      unsigned int tX,  unsigned int tY  ,
                      unsigned int sX,  unsigned int sY  ,
                      unsigned int patchWidth , unsigned int patchHeight
                     )
{
    /*
  //Check for bounds -----------------------------------------
  if (tX+width>=targetWidth) { width=targetWidth-tX-1;  }
  if (tY+height>=targetHeight) { height=targetHeight-tY-1;  }
  //----------------------------------------------------------

  unsigned char * targetPTR; unsigned char * targetLineLimitPTR; unsigned char * targetLimitPTR;   unsigned int targetLineSkip;
  targetPTR      = target + MEMPLACE3(tX,tY,targetWidth);
  targetLimitPTR = target + MEMPLACE3((tX+width),(tY+height),targetWidth);
  targetLineSkip = (targetWidth-width) * 3;
  targetLineLimitPTR = targetPTR + (width*3) -3; //-3 is required here

  fprintf(stderr,"BitBlt Color an area (%u,%u) of target image  starting at %u,%u  sized %u,%u with color RGB(%u,%u,%u)\n",width,height,tX,tY,targetWidth,targetHeight,R,G,B);
  fprintf(stderr,"last Pixels @ %u,%u\n",tX+width,tY+height);
  while ( targetPTR < targetLimitPTR )
  {
     while (targetPTR < targetLineLimitPTR)
     {
        //fprintf(stderr,"Reading Triplet sourcePTR %p targetPTR is %p\n",sourcePTR  ,targetPTR);
        *targetPTR = R; ++targetPTR;
        *targetPTR = G; ++targetPTR;
        *targetPTR = B; ++targetPTR;
     }
    targetLineLimitPTR += targetWidth*3;
    targetPTR+=targetLineSkip;
  }
  */
 return 1;
}
