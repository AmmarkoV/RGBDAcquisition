#include "dericheRecursiveGaussian.h"
#include "constantTimeBilateralFiltering.h"
#include "../tools/imageMatrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>


struct ctbfPool
{
  float * Wk;
  float * Jk;
  unsigned char * JBk;
};

#define SR  0.006
#define SR2 2*SR
inline float wKResponseF(float kBin , float in , float divider )
{
  float response = -1 * ( in - kBin )  * ( in - kBin );
  response = response / (SR2);
  return (float) exp(response)/divider;
}


inline float wKResponseUC(float kBin , unsigned char in , float divider )
{
  float response = -1 * ( in - kBin )  * ( in - kBin );
  response = response / (SR2);
  return (float) exp(response)/divider;
}


void populateWKMatrix( float * Wk , float k , float divider , unsigned char * source , unsigned int sourceWidth, unsigned int sourceHeight  )
{
  float *WkPTR=Wk;
  unsigned char *  sourcePTR=source , sourceLimit = source + sourceWidth * sourceHeight;

  while (sourcePTR < sourceLimit)
  {
     *WkPTR = (float)  wKResponseUC(k , *sourcePTR , divider );
     ++WkPTR; ++sourcePTR;
  }
}



void populateJKMatrix( float * Jk , float * Wk  , unsigned char * source , unsigned int sourceWidth, unsigned int sourceHeight  )
{
   multiply2DMatricesFWithUC(Jk,Wk,source,sourceWidth,sourceHeight);
}

void populateJBkMatrix( float * JBk ,  float * der1 , float * der2  , unsigned int sourceWidth, unsigned int sourceHeight )
{
  divide2DMatricesF(JBk, der1 , der2 , sourceWidth , sourceHeight );
}

int constantTimeBilateralFilter(
                                unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int channels ,
                                unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                                float sigma ,
                                unsigned int bins
                               )
{
  unsigned int doNormalization = 0;

  float * tmp1 =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight );
  if (tmp1==0) { return 0; }
  float * tmp2 =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight );
  if (tmp2==0) { free(tmp1); return 0; }

  unsigned int i=0;
  struct ctbfPool * ctfbp = (struct ctbfPool *) malloc( sizeof(struct ctbfPool) * bins );
  if (ctfbp == 0 ) { free(tmp1); free(tmp2); return 0; }

  for (i=0; i<bins; i++)
  {
    ctfbp[i].Wk =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight );
    ctfbp[i].Jk =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight );
  }

  float maxVal=255;
  float step = maxVal / ( bins-1);
  float divider = sqrt( M_PI *(SR2));

  unsigned int quantizationStep  =  (unsigned int) maxVal / step;
  unsigned int k=0;

for (i=0; i<bins; i++)
{

 populateWKMatrix( ctfbp[i].Wk , k , divider , source , sourceWidth , sourceHeight  );
 populateJKMatrix( ctfbp[i].Jk , ctfbp[i].Wk  , source , sourceWidth, sourceHeight  );


 dericheRecursiveGaussianGray(
                              ctfbp[i].Jk,  sourceWidth , sourceHeight , channels,
                              tmp1,  targetWidth , targetHeight ,
                              sigma , 0
                             );


 dericheRecursiveGaussianGray(
                              ctfbp[i].Wk,  sourceWidth , sourceHeight , channels,
                              tmp2,  targetWidth , targetHeight ,
                              sigma , 0
                             );


  populateJBkMatrix(ctfbp[i].JBk ,  ctfbp[i].Jk , ctfbp[i].Wk  , sourceWidth, sourceHeight );



 k+=quantizationStep;
}


//TODO : Store Result on target




//Deallocated everything that is useless
  for (i=0; i<bins; i++)
  {
    free( ctfbp[i].Wk  );
    free( ctfbp[i].Jk  );
    free( ctfbp[i].JBk );
  }
 free(ctfbp);
 free(tmp1);
 free(tmp2);
 return 1;
}


