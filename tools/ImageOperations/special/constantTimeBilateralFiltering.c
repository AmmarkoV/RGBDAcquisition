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
  float * JBk;
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
  unsigned char *  sourcePTR=source , * sourceLimit = source + sourceWidth * sourceHeight;

  while (sourcePTR < sourceLimit)
  {
     *WkPTR = (float)  wKResponseUC(k , *sourcePTR , divider );
     ++WkPTR; ++sourcePTR;
  }
}



void populateJKMatrix( float * Jk , float * Wk  , unsigned char * source , unsigned int sourceWidth, unsigned int sourceHeight  )
{
   multiply2DMatricesFWithUC(Jk,Wk,source,sourceWidth,sourceHeight,1);
}

void populateJBkMatrix( float * JBk ,  float * der1 , float * der2  , unsigned int sourceWidth, unsigned int sourceHeight )
{
  divide2DMatricesF(JBk, der1 , der2 , sourceWidth , sourceHeight ,1);
}





int constantTimeBilateralFilter(
                                unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int channels ,
                                unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                                float sigma ,
                                unsigned int bins
                               )
{
  if (channels!=1)
  {
    fprintf(stderr,"constantTimeBilateralFilter cannot work with more than 1 channels");
    return 0;
  }
  unsigned int doNormalization = 0;

  float * tmp1 =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight * channels);
  if (tmp1==0) { return 0; }
  float * tmp2 =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight * channels);
  if (tmp2==0) { free(tmp1); return 0; }

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
  float divider = sqrt( M_PI *(SR2));

  unsigned int quantizationStep  =  (unsigned int) maxVal / step;
  unsigned int k=0;

fprintf(stderr,"Making bins : ");
for (i=0; i<bins; i++)
{
 fprintf(stderr,".");
 //fprintf(stderr,"wk");
 populateWKMatrix( ctfbp[i].Wk , k , divider , source , sourceWidth , sourceHeight  );
 //fprintf(stderr,"jk");
 populateJKMatrix( ctfbp[i].Jk , ctfbp[i].Wk  , source , sourceWidth, sourceHeight  );


 //fprintf(stderr,"2xder");
 dericheRecursiveGaussianGrayF(
                              ctfbp[i].Jk,  sourceWidth , sourceHeight , channels,
                              tmp1,  targetWidth , targetHeight ,
                              sigma , 0
                             );


 dericheRecursiveGaussianGrayF(
                              ctfbp[i].Wk,  sourceWidth , sourceHeight , channels,
                              tmp2,  targetWidth , targetHeight ,
                              sigma , 0
                             );


 //fprintf(stderr,"jbk");
 populateJBkMatrix(ctfbp[i].JBk ,  tmp1 , tmp2  , sourceWidth, sourceHeight );



 k+=quantizationStep;
}
 fprintf(stderr,"\n");


//TODO : Store Result on target
fprintf(stderr,"Collecting result \n");

unsigned char *inVal ;
unsigned char *resPTR = target;
unsigned char *resLimit = target+targetWidth*targetHeight;
float * jbkPTR;
//float q_ceil , q_floor;
unsigned int x=0,y=0;

while (resPTR<resLimit)
{
 inVal = source + ( y * sourceWidth ) + x;
 i =  *inVal/step;


 //fprintf(stderr,"i(%u,%u)=%u[%u/%u] ",x,y,*inVal,i,bins);
 //DO INTERPOLATION..
 //q_ceil=ceil(val/D);
 //q_floor=floor(val/D);
 //a=(q_ceil*D-val)/D;
 //tmp=(a)*JBk(i,j,q_floor+1)+(1-a)*JBk(i,j,q_ceil+1);
 jbkPTR = ctfbp[i].JBk + ( y * targetWidth ) + x;
 float outIntensity = *jbkPTR * step;
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
 if (tmp1!=0 ) { free(tmp1); }
 if (tmp2!=0 ) { free(tmp2); }
 if (ctfbp!=0 ) { free(ctfbp); }
 return 1;
}


