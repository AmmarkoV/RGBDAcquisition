#include "imageMatrix.h"
#include <stdio.h>
#include <stdlib.h>



int castUCharImage2FloatAndNormalize(float * out , unsigned char * in, unsigned int width,unsigned int height , unsigned int channels)
{
  float *outPTR = out;
  float *outLimit = out + ( width * height * channels);
  unsigned char *inPTR = in;

  if (out!=0)
   {
     while(outPTR<outLimit)
     {
       *outPTR = (float) *inPTR/255;
       ++outPTR;
       ++inPTR;
     }
    return 1;
   }
  return 0;
}


int castUCharImage2Float(float * out , unsigned char * in, unsigned int width,unsigned int height , unsigned int channels)
{
  float *outPTR = out;
  float *outLimit = out + ( width * height * channels);
  unsigned char *inPTR = in;

  if (out!=0)
   {
     while(outPTR<outLimit)
     {
       *outPTR = (float) *inPTR;
       ++outPTR;
       ++inPTR;
     }
    return 1;
   }
  return 0;
}

int castFloatImage2UChar(unsigned char * out , float * in, unsigned int width,unsigned int height , unsigned int channels)
{
  if ( (out==0) || (in==0) )
    {
      return 0;
    }

  unsigned char *outPTR = out;
  unsigned char *outLimit = out + ( width * height * channels);
  float *inPTR = in;

     while(outPTR<outLimit)
     {
       *outPTR = (unsigned char) *inPTR;
       ++outPTR;
       ++inPTR;
     }
    return 1;
}


float * copyUCharImage2Float(unsigned char * in, unsigned int width,unsigned int height , unsigned int channels)
{
  float * out  = ( float * ) malloc(sizeof(float) * width * height * channels);
  if (out!=0)
  {
  if (castUCharImage2Float(out , in, width, height , channels) )
    {
       return out;
    }
   free(out);
  }
  return 0;
}

/*  dividend / divisor = quotient */
unsigned char* divideTwoImages(unsigned char *  dividend , unsigned char * divisor , unsigned int width,unsigned int height , unsigned int channels)
{
  unsigned char *res = (unsigned char*) malloc(sizeof(unsigned char) * width * height * channels );

  unsigned char *dividendPTR = dividend;
  unsigned char *divisorPTR = divisor;
  unsigned char *resPTR = res;
  unsigned char *resLimit = res+width*height * channels;
  if (res!=0)
   {
     while(resPTR<resLimit)
     {
       *resPTR = (unsigned char) *dividendPTR / *divisorPTR;
       ++resPTR;
       ++divisorPTR;
       ++dividendPTR;
     }
   }
  return res;
}

/*  dividend / divisor = quotient */
int divide2DMatricesF(float * out , float * dividend , float * divisor , unsigned int width , unsigned int height , unsigned int channels)
{
  if ( (out==0)||(out==dividend)||(out==divisor) ) { return 0; }
  float *dividendPTR=dividend , *divisorPTR=divisor , *outPTR=out , *outLimit=out + (width * height * channels);

  while (outPTR < outLimit)
  {
     if (*divisorPTR!=0) { *outPTR = (float) (*dividendPTR) / (*divisorPTR) ;  } else
                         { *outPTR=0; }
     ++divisorPTR; ++dividendPTR; ++outPTR;
  }
 return 1;
}


int multiply2DMatricesF(float * out , float * mult1 , float * mult2 , unsigned int width , unsigned int height , unsigned int channels)
{
  float *mult1PTR=mult1 , *mult2PTR=mult2 , *outPTR=out , *outLimit=out + (width * height * channels);

  while (outPTR < outLimit)
  {
     *outPTR = (float) (*mult1PTR) * (* mult2PTR) ;
     ++mult1PTR; ++mult2PTR; ++outPTR;
  }
 return 1;
}

int multiply2DMatricesFWithUC(float * out , float * mult1 , unsigned char * mult2 , unsigned int width , unsigned int height , unsigned int channels )
{
  float *mult1PTR=mult1 , *outPTR=outPTR , *outLimit=outPTR + (width * height * channels);
  unsigned char *  mult2PTR=mult2;

  while (outPTR < outLimit)
  {
     *outPTR = (float) (*mult1PTR) * (* mult2PTR) ;
     ++mult1PTR; ++mult2PTR; ++outPTR;
  }
 return 1;
}

