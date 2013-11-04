#include "patchComparison.h"
#include <stdio.h>

#define MEMPLACE3(x,y,width) ( y * ( width * 3 ) + x*3 )
#define MEMPLACE1(x,y,width) ( y * ( width ) + x )
#define RGB(r,g,b) B + (G * 256) + (R * 65536) )
#define ABSDIFF(num1,num2) ( (num1-num2) >=0 ? (num1-num2) : (num2 - num1) )


#define COMPARISON_MODE 1


unsigned int compareDepthPatches( unsigned short * patchADepth , unsigned int pACenterX,  unsigned int pACenterY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                                  unsigned short * patchBDepth , unsigned int pBCenterX,  unsigned int pBCenterY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                                  unsigned int patchWidth, unsigned int patchHeight )
{
  if ( (patchADepth==0)||(patchBDepth==0) ) { return 0; }
  if ( (patchWidth==0)&&(patchWidth==0) ) { return 0; }
  if ( (pAImageWidth==0)&&(pAImageHeight==0) ) { return 0; }
  if ( (pBImageWidth==0)&&(pBImageHeight==0) ) { return 0; }


  unsigned int halfWidth = (unsigned int) patchWidth / 2 ;
  unsigned int halfHeight = (unsigned int) patchHeight / 2 ;


  unsigned int pACX = pACenterX - halfWidth;
  if (halfWidth>pACenterX) { pACX=0; }
  unsigned int pACY = pACenterY - halfHeight;
  if (halfHeight>pACenterY) { pACY=0; }

  unsigned int pBCX = pBCenterX - halfWidth;
  if (halfWidth>pBCenterX) { pBCX=0; }
  unsigned int pBCY = pBCenterY - halfHeight;
  if (halfHeight>pBCenterY) { pBCY=0; }

  //Check for bounds -----------------------------------------
  if (pBCX+patchWidth>=pBImageWidth)   { patchWidth=pBImageWidth-pBCX;  }
  if (pBCY+patchHeight>=pBImageHeight) { patchHeight=pBImageHeight-pBCY;  }

  if (pACX+patchWidth>=pAImageWidth)   { patchWidth=pAImageWidth-pACX;  }
  if (pACY+patchHeight>=pAImageHeight) { patchHeight=pAImageHeight-pACY;  }
  //----------------------------------------------------------





  fprintf(stderr,"compareDepthPatches ( %u,%u -> %u,%u ) vs ( %u,%u -> %u,%u ) \n",pACX,pACY,pACX+patchWidth,pACY+patchHeight,
                                                                                   pBCX,pBCY,pBCX+patchWidth,pBCY+patchHeight);



  unsigned short * pA_PTR      = patchADepth+ MEMPLACE1(pACX,pACY,pAImageWidth);
  unsigned short * pA_LimitPTR = patchADepth+ MEMPLACE1((pACX+patchWidth),(pACY+patchHeight),pAImageWidth);
  unsigned int pA_LineSkip = (pAImageWidth-patchWidth) ;
  unsigned short * pA_LineLimitPTR = pA_PTR + (patchWidth);

  unsigned short * pB_PTR      = patchBDepth + MEMPLACE1(pBCX,pBCY,pBImageWidth);
  unsigned short * pB_LimitPTR = patchBDepth + MEMPLACE1((pBCX+patchWidth),(pBCY+patchHeight),pBImageWidth); // <- IS THIS OK ? or are they *2
  unsigned int pB_LineSkip = (pBImageWidth-patchWidth) ;

  unsigned int score = 10000;
  unsigned int penalty = 10;

  while (pA_PTR < pA_LimitPTR)
  {
     while (pA_PTR < pA_LineLimitPTR)
     {

         #if   COMPARISON_MODE == 1
          score+=ABSDIFF((*pA_PTR),(*pB_PTR));
         #elif COMPARISON_MODE == 2
           if ( (*pA_PTR>0) && (*pB_PTR>0) ) {
                                               if (score > penalty) {  score-=penalty ; }
                                             } else
                                             { ++score; }
         #endif // COMPARISON_MODE

         ++pA_PTR;
         ++pB_PTR;
     }
    pA_LineLimitPTR+= pAImageWidth;
    pA_PTR+=pA_LineSkip;
    pB_PTR+=pB_LineSkip;
  }


  return score;
}



