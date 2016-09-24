#include "dericheRecursiveGaussian.h"
#include "constantTimeBilateralFiltering.h"
#include "../tools/imageMatrix.h"
#include "../imageFilters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

struct ctbfPool
{
  float * Wk;
  float * Jk;
  float * JBk;
};

#define SR  0.006
#define SR_MUL_2 2*SR
static inline float wKResponse(float kBin , float * in , float divider )
{
  float inMinusK = *in - kBin;
  float response = -1 * inMinusK  * inMinusK;
  response = response / (SR_MUL_2);

  return (float) exp(response)/divider;
}


void populateWKMatrix( float * Wk , float k , float divider , float * source , unsigned int sourceWidth, unsigned int sourceHeight  )
{
  float *WkPTR=Wk;
  float *sourcePTR=source , *sourceLimit = source + sourceWidth * sourceHeight;

  while (sourcePTR < sourceLimit)
  {
     *WkPTR = (float)  wKResponse(k , sourcePTR , divider );
     ++WkPTR; ++sourcePTR;
  }
}



void populateJKMatrix( float * Jk , float * Wk  , float * source , unsigned int sourceWidth, unsigned int sourceHeight  )
{
   multiply2DMatricesF(Jk,Wk,source,sourceWidth,sourceHeight,1);
}

void populateJBkMatrix( float * JBk ,  float * der1 , float * der2  , unsigned int sourceWidth, unsigned int sourceHeight )
{
  divide2DMatricesF(JBk, der1 , der2 , sourceWidth , sourceHeight ,1);
}





int constantTimeBilateralFilter(
                                unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int channels ,
                                unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                                float * sigma ,
                                unsigned int bins ,
                                int useDeriche
                               )
{
  if (channels!=1)
  {
    fprintf(stderr,"constantTimeBilateralFilter cannot work with more than 1 channels\n");
    return 0;
  }

 if (
      (targetWidth!=sourceWidth)  ||
      (targetHeight!=sourceHeight)
    )
 {
    fprintf(stderr,"constantTimeBilateralFilter cannot work with different size images\n");
    return 0;
 }

 float * convolutionMatrix=0;
 unsigned int kernelWidth=21;
 unsigned int kernelHeight=kernelWidth;
 float divisor=1.0;

 if (useDeriche==0)
 {
    convolutionMatrix = allocateGaussianKernel(kernelWidth,*sigma,1);
 }


  float * sourceNormalized =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight * channels);
  if (sourceNormalized==0) { return 0; }

  castUCharImage2FloatAndNormalize(sourceNormalized,source,sourceWidth,sourceHeight,channels);

  float * tmp1 =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight * channels);
  if (tmp1==0) { free(sourceNormalized); return 0; }
  float * tmp2 =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight * channels);
  if (tmp2==0) { free(sourceNormalized); free(tmp1); return 0; }

  unsigned int i=0;
  struct ctbfPool * ctfbp = (struct ctbfPool *) malloc( sizeof(struct ctbfPool) * bins );
  if (ctfbp == 0 ) { free(tmp1); free(tmp2); return 0; }

  for (i=0; i<bins; i++)
  {
    ctfbp[i].Wk =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight * channels);
    if (ctfbp[i].Wk==0) { fprintf(stderr,"Could not allocate a wK buffer\n"); }

    ctfbp[i].Jk =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight * channels);
    if (ctfbp[i].Jk==0) { fprintf(stderr,"Could not allocate a jK buffer\n"); }

    ctfbp[i].JBk =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight * channels);
    if (ctfbp[i].JBk==0) { fprintf(stderr,"Could not allocate a jK buffer\n"); }
  }

  float maxVal=255;
  float step = maxVal / ( bins-1);
  float divider = sqrt( M_PI *(SR_MUL_2));

  unsigned int quantizationStep  =  (unsigned int) maxVal / step;
  unsigned int k=0;

fprintf(stderr,"Making bins : ");
for (i=0; i<bins; i++)
{
 fprintf(stderr,".");
 //fprintf(stderr,"wk");
 populateWKMatrix( ctfbp[i].Wk , (float) k / maxVal , divider , sourceNormalized , sourceWidth , sourceHeight  );
 //fprintf(stderr,"jk");
 populateJKMatrix( ctfbp[i].Jk , ctfbp[i].Wk  , sourceNormalized , sourceWidth, sourceHeight  );


 //fprintf(stderr,"2xder");

 if (useDeriche==0)
 {
   convolutionFilter1ChF(
                          tmp1,  targetWidth , targetHeight ,
                          ctfbp[i].Jk,  sourceWidth , sourceHeight  ,
                          convolutionMatrix , kernelWidth , kernelHeight , divisor
                         );


   convolutionFilter1ChF(
                          tmp2,  targetWidth , targetHeight ,
                          ctfbp[i].Wk,  sourceWidth , sourceHeight  ,
                          convolutionMatrix , kernelWidth , kernelHeight , divisor
                         );
 } else
 if (useDeriche==1)
 {
 dericheRecursiveGaussianGrayF(
                                tmp1,  targetWidth , targetHeight ,channels,
                                ctfbp[i].Jk,  sourceWidth , sourceHeight ,
                                sigma , 0
                              );


 dericheRecursiveGaussianGrayF(
                                tmp2,  targetWidth , targetHeight , channels,
                                ctfbp[i].Wk,  sourceWidth , sourceHeight ,
                                sigma , 0
                              );
 } else
 {
    memcpy(tmp1,ctfbp[i].Jk,sourceWidth*sourceHeight*channels*sizeof(float));
    memcpy(tmp2,ctfbp[i].Wk,sourceWidth*sourceHeight*channels*sizeof(float));
 }


 //fprintf(stderr,"jbk");
 populateJBkMatrix(ctfbp[i].JBk ,  tmp1 , tmp2  , sourceWidth, sourceHeight );


 k+=quantizationStep;
}
 fprintf(stderr,"\n");


fprintf(stderr,"Collecting result ..\n");
unsigned char *inVal;
unsigned char *resPTR = target;
unsigned char *resLimit = target+targetWidth*targetHeight;
float * jbkPTRMin;
float * jbkPTRMax;

unsigned int x=0,y=0;
unsigned int iMin,iMax;
float a;

while (resPTR<resLimit)
{
 inVal = source + ( y * sourceWidth ) + x;
 i =  *inVal/step;
 iMin = (unsigned int) ceil ((float) *inVal/step );
 iMax = (unsigned int) floor((float) *inVal/step );

 a=(float) ( iMin*step - *inVal ) / step;

 jbkPTRMin = ctfbp[iMin].JBk + ( y * targetWidth ) + x;
 jbkPTRMax = ctfbp[iMax].JBk + ( y * targetWidth ) + x;
 float outIntensity = (a * (*jbkPTRMax)) + ((1-a) * (*jbkPTRMin));
 outIntensity = outIntensity * maxVal; // Go back from normalized image

 *resPTR = (unsigned char) outIntensity;

 ++x;
 if (x>=targetWidth) { x=0; ++y; }
 ++resPTR;
}


fprintf(stderr,"Deallocating everything \n");
//Deallocated everything that is useless
  for (i=0; i<bins; i++)
  {
    if (ctfbp[i].Wk!=0 )  { free( ctfbp[i].Wk ); }
    if (ctfbp[i].Jk!=0 )  { free( ctfbp[i].Jk  ); }
    if (ctfbp[i].JBk!=0 ) { free( ctfbp[i].JBk ); }
  }

 if (convolutionMatrix!=0) { free(convolutionMatrix); }
 if (sourceNormalized !=0 ) { free(sourceNormalized ); }
 if (tmp1!=0 ) { free(tmp1); }
 if (tmp2!=0 ) { free(tmp2); }
 if (ctfbp!=0 ) { free(ctfbp); }
 return 1;
}


