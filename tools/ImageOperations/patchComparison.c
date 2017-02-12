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
  if ( (patchWidth==0) ||(patchHeight==0) ) { return 0; }
  if ( (pAImageWidth==0)||(pAImageHeight==0) ) { return 0; }
  if ( (pBImageWidth==0)||(pBImageHeight==0) ) { return 0; }


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
  //unsigned short * pB_LimitPTR = patchBDepth + MEMPLACE1((pBCX+patchWidth),(pBCY+patchHeight),pBImageWidth); // <- IS THIS OK ? or are they *2
  unsigned int pB_LineSkip = (pBImageWidth-patchWidth) ;

  unsigned int score = 10000;

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




int compareRGBPatches( unsigned char * patchARGB , unsigned int pACX,  unsigned int pACY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                       unsigned char * patchBRGB , unsigned int pBCX,  unsigned int pBCY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                       unsigned int patchWidth, unsigned int patchHeight  ,
                       unsigned int * score
                     )
{
  if ( (patchARGB==0)||(patchBRGB==0) ) { return 0; }
  if ( (patchWidth==0)||(patchHeight==0) ) { return 0; }
  if ( (pAImageWidth==0)||(pAImageHeight==0) ) { return 0; }
  if ( (pBImageWidth==0)||(pBImageHeight==0) ) { return 0; }
  if ( (pACX>=pAImageWidth)||(pACY>=pAImageHeight) ) { return 0; }
  if ( (pBCX>=pBImageWidth)||(pBCY>=pBImageHeight) ) { return 0; }


  //Check for bounds -----------------------------------------
  if (pACX+patchWidth>=pAImageWidth)   { patchWidth=pAImageWidth-pACX;  }
  if (pACY+patchHeight>=pAImageHeight) { patchHeight=pAImageHeight-pACY;  }

  if (pBCX+patchWidth>=pBImageWidth)   { patchWidth=pBImageWidth-pBCX;  }
  if (pBCY+patchHeight>=pBImageHeight) { patchHeight=pBImageHeight-pBCY;  }

  if ( (patchWidth==0)||(patchHeight==0) ) { return 0; }
  //----------------------------------------------------------

  //fprintf(stderr,"imageA ( %u,%u ) - imageB ( %u,%u ) \n",pAImageWidth,pAImageHeight,pBImageWidth,pBImageHeight);

  fprintf(stderr,"compareRGBPatches ( %u,%u -> %u,%u ) vs ( %u,%u -> %u,%u )  patch %u,%u \n",pACX,pACY,pACX+patchWidth,pACY+patchHeight,
                                                                                              pBCX,pBCY,pBCX+patchWidth,pBCY+patchHeight,
                                                                                              patchWidth,patchHeight
                                                                                              );

  unsigned char * pA_PTR      = patchARGB+ MEMPLACE3(pACX,pACY,pAImageWidth);
  unsigned char * pA_LimitPTR = patchARGB+ MEMPLACE3((pACX+patchWidth),(pACY+patchHeight),pAImageWidth);
  unsigned int pA_LineSkip = (pAImageWidth-patchWidth)*3 ;
  unsigned char * pA_LineLimitPTR = pA_PTR + (patchWidth*3);

  unsigned char * pB_PTR      = patchBRGB + MEMPLACE3(pBCX,pBCY,pBImageWidth);
  //unsigned char * pB_LimitPTR = patchBRGB + MEMPLACE3((pBCX+patchWidth),(pBCY+patchHeight),pBImageWidth); // <- IS THIS OK ? or are they *2
  unsigned int pB_LineSkip = (pBImageWidth-patchWidth)*3 ;

  *score = 0;

  //fprintf(stderr,"pA_PTR = %u , pA_LimitPTR = %u , pA_LineSkip = %u , pA_LineLimitPTR = %u \n",pA_PTR,pA_LimitPTR,pA_LineSkip,pA_LineLimitPTR);
  //fprintf(stderr,"pB_PTR = %u , pB_LineSkip = %u \n",pB_PTR,pB_LineSkip);


  while (pA_PTR < pA_LimitPTR)
  {
     while (pA_PTR < pA_LineLimitPTR)
     {
          *score+=(unsigned int) ABSDIFF((*pA_PTR),(*pB_PTR));   ++pA_PTR;  ++pB_PTR;
          *score+=(unsigned int) ABSDIFF((*pA_PTR),(*pB_PTR));   ++pA_PTR;  ++pB_PTR;
          *score+=(unsigned int) ABSDIFF((*pA_PTR),(*pB_PTR));   ++pA_PTR;  ++pB_PTR;
     }
    pA_LineLimitPTR+= pAImageWidth*3;
    pA_PTR+=pA_LineSkip;
    pB_PTR+=pB_LineSkip;
  }

  fprintf(stderr,"Result = %u \n",*score);
  return 1;
}



int compareRGBPatchesIgnoreColor
                     ( unsigned char * patchARGB , unsigned int pACX,  unsigned int pACY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                       unsigned char * patchBRGB , unsigned int pBCX,  unsigned int pBCY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                       unsigned char ignoreR , unsigned char ignoreG , unsigned char ignoreB ,
                       unsigned int patchWidth, unsigned int patchHeight  ,
                       unsigned int * score,
                       unsigned int  failScore
                     )
{
  if ( (patchARGB==0)||(patchBRGB==0) ) { return 0; }
  if ( (patchWidth==0)||(patchHeight==0) ) { return 0; }
  if ( (pAImageWidth==0)||(pAImageHeight==0) ) { return 0; }
  if ( (pBImageWidth==0)||(pBImageHeight==0) ) { return 0; }
  if ( (pACX>=pAImageWidth)||(pACY>=pAImageHeight) ) { return 0; }
  if ( (pBCX>=pBImageWidth)||(pBCY>=pBImageHeight) ) { return 0; }


  //Check for bounds -----------------------------------------
  if (pACX+patchWidth>=pAImageWidth)   { patchWidth=pAImageWidth-pACX;  }
  if (pACY+patchHeight>=pAImageHeight) { patchHeight=pAImageHeight-pACY;  }

  if (pBCX+patchWidth>=pBImageWidth)   { patchWidth=pBImageWidth-pBCX;  }
  if (pBCY+patchHeight>=pBImageHeight) { patchHeight=pBImageHeight-pBCY;  }

  if ( (patchWidth==0)||(patchHeight==0) ) { return 0; }
  //----------------------------------------------------------

  unsigned char * pA_PTR      = patchARGB+ MEMPLACE3(pACX,pACY,pAImageWidth);
  unsigned char * pA_LimitPTR = patchARGB+ MEMPLACE3((pACX+patchWidth),(pACY+patchHeight),pAImageWidth);
  unsigned int pA_LineSkip = (pAImageWidth-patchWidth)*3 ;
  unsigned char * pA_LineLimitPTR = pA_PTR + (patchWidth*3);

  unsigned char * pB_PTR      = patchBRGB + MEMPLACE3(pBCX,pBCY,pBImageWidth);
  unsigned int pB_LineSkip = (pBImageWidth-patchWidth)*3 ;

  unsigned int tmpScore = 0;


  unsigned char R1,G1,B1,R2,G2,B2;
  unsigned int ignored=0;

  while (pA_PTR < pA_LimitPTR)
  {
   //fprintf(stderr,"pA_PTR = %u , pA_LimitPTR = %u , pA_LineSkip = %u , pA_LineLimitPTR = %u \n",pA_PTR,pA_LimitPTR,pA_LineSkip,pA_LineLimitPTR);
   //fprintf(stderr,"pB_PTR = %u , pB_LineSkip = %u \n",pB_PTR,pB_LineSkip);

     while (pA_PTR < pA_LineLimitPTR)
     {
         R1 = (*pA_PTR); ++pA_PTR;  R2 = (*pB_PTR); ++pB_PTR;
         G1 = (*pA_PTR); ++pA_PTR;  G2 = (*pB_PTR); ++pB_PTR;
         B1 = (*pA_PTR); ++pA_PTR;  B2 = (*pB_PTR); ++pB_PTR;

         if ( (R1==ignoreR) && (G1==ignoreG) && (B1==ignoreB) ) { ++ignored; /*Haystack patch pixel is transparent*/ } else
         if ( (R2==ignoreR) && (G2==ignoreG) && (B2==ignoreB) ) { ++ignored; /*Needle patch pixel is transparent*/ }   else
         {
          tmpScore+=(unsigned int) ABSDIFF(R1,R2);
          tmpScore+=(unsigned int) ABSDIFF(G1,G2);
          tmpScore+=(unsigned int) ABSDIFF(B1,B2);
         }
     }

    if (tmpScore>failScore+1) { *score=tmpScore; return 1; }

    pA_LineLimitPTR+= pAImageWidth*3;
    pA_PTR+=pA_LineSkip;
    pB_PTR+=pB_LineSkip;
  }

  *score = tmpScore;

  //fprintf(stderr,"Result = %u , Ignored = %u\n",*score,ignored);
  return 1;
}


int compareRGBPatchesNeighborhoodIgnoreColor
                     (
                       unsigned int neighborhoodX , unsigned int neighborhoodY ,
                       unsigned char * patchARGB , unsigned int pACX,  unsigned int pACY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                       unsigned char * patchBRGB , unsigned int pBCX,  unsigned int pBCY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                       unsigned char ignoreR , unsigned char ignoreG , unsigned char ignoreB ,
                       unsigned int patchWidth, unsigned int patchHeight  ,
                       unsigned int * score ,
                       unsigned int  failScore
                     )
{

    unsigned int xS,yS;
    xS=pACX;  if (xS>neighborhoodX) { xS=xS-neighborhoodX; } else { xS=0; }
    yS=pACY;  if (yS>neighborhoodY) { yS=yS-neighborhoodY; } else { yS=0; }

    unsigned int xE,yE;
    xE=pACX;  if (xE+neighborhoodX<pAImageWidth )  { xE=xE+neighborhoodX; }
    yE=pACY;  if (yE+neighborhoodY<pAImageHeight ) { yE=yE+neighborhoodY; }

    unsigned int bestScore=failScore+1;
    unsigned int currentScore=0;

    unsigned int x,y;
    for (y=yS; y<yE; y++)
    {
      for (x=xS; x<xE; x++)
       {
           compareRGBPatchesIgnoreColor
                     ( patchARGB , x,  y , pAImageWidth , pAImageHeight ,
                       patchBRGB , pBCX,  pBCY , pBImageWidth , pBImageHeight ,
                       ignoreR , ignoreG , ignoreB ,
                       patchWidth, patchHeight  ,
                       &currentScore ,
                       failScore
                     );
            if (currentScore<bestScore) { bestScore=currentScore; }
       }
    }

   *score=bestScore;
   return (bestScore<failScore);
}




unsigned int colorVariance( unsigned char * pixels , unsigned int imageWidth ,unsigned int imageHeight ,
                            unsigned int pX,  unsigned int pY, unsigned int width , unsigned int height)
{
  if (pixels==0) { return 0; }
  if ( (imageWidth==0) && (imageHeight==0) ) { return 0; }

  unsigned char * pA_PTR      = pixels + MEMPLACE3(pX,pY,imageWidth);
  unsigned char * pA_LimitPTR = pixels + MEMPLACE3((pX+width),(pY+height),imageWidth);
  unsigned int pA_LineSkip = (imageWidth-width)*3 ;
  unsigned char * pA_LineLimitPTR = pA_PTR + (width*3);


  unsigned char R = 0 , G = 0 , B = 0;
  unsigned char lastR = 0 , lastG = 0 , lastB = 0;
  unsigned char * pTmp_PTR = pA_PTR;
  lastR = *pTmp_PTR; ++pTmp_PTR;
  lastG = *pTmp_PTR; ++pTmp_PTR;
  lastB = *pTmp_PTR; ++pTmp_PTR;

//  signed int thisScore=0;
  unsigned int score=0;

  while (pA_PTR < pA_LimitPTR)
  {
     while (pA_PTR < pA_LineLimitPTR)
     {
        R = * pA_PTR; ++pA_PTR;
        G = * pA_PTR; ++pA_PTR;
        B = * pA_PTR; ++pA_PTR;

        if (R>=lastR) { score+=R-lastR; } else { score+=lastR-R; }
        if (G>=lastG) { score+=G-lastG; } else { score+=lastG-G; }
        if (B>=lastB) { score+=B-lastB; } else { score+=lastB-B; }

        lastR=R; lastG=G; lastB=B;
     }

    pA_LineLimitPTR+= imageWidth*3;
    pA_PTR+=pA_LineSkip;
  }

  //fprintf(stderr,"colorVariance(%u,%u,%u,%u) = %u \n", pX,   pY,  width , height , score);
  return score;
}





int compareRGBPatchesCenter( unsigned char * patchARGB , unsigned int pACenterX,  unsigned int pACenterY , unsigned int pAImageWidth , unsigned int pAImageHeight ,
                             unsigned char * patchBRGB , unsigned int pBCenterX,  unsigned int pBCenterY , unsigned int pBImageWidth , unsigned int pBImageHeight ,
                             unsigned int patchWidth, unsigned int patchHeight  ,
                             unsigned int * score )
{
  if ( (patchARGB==0)||(patchBRGB==0) )        { return 0; }
  if ( (patchWidth==0)||(patchHeight==0) )      { return 0; }
  if ( (pAImageWidth==0)||(pAImageHeight==0) ) { return 0; }
  if ( (pBImageWidth==0)||(pBImageHeight==0) ) { return 0; }


  unsigned int AhalfWidth  = (unsigned int) pAImageWidth / 2 ;
  unsigned int AhalfHeight = (unsigned int) pAImageHeight / 2 ;

  unsigned int BhalfWidth  = (unsigned int) pBImageWidth / 2 ;
  unsigned int BhalfHeight = (unsigned int) pBImageHeight / 2 ;


  return compareRGBPatchesCenter( patchARGB, pACenterX-AhalfWidth  ,  pACenterY-AhalfHeight , pAImageWidth , pAImageHeight ,
                                  patchBRGB, pBCenterX-BhalfWidth  ,  pBCenterY-BhalfHeight , pBImageWidth , pBImageHeight ,
                                  patchWidth, patchHeight , score);
}

