#include "imageMatrix.h"
#include <stdio.h>
#include <stdlib.h>



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
  unsigned char *outPTR = out;
  unsigned char *outLimit = out + ( width * height * channels);
  float *inPTR = in;

  if (out!=0)
   {
     while(outPTR<outLimit)
     {
       *outPTR = (unsigned char) *inPTR;
       ++outPTR;
       ++inPTR;
     }
    return 1;
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
  float *dividendPTR=dividend , *divisorPTR=divisor , *outPTR=outPTR , *outLimit=outPTR + (width * height * channels);

  while (outPTR < outLimit)
  {
     *outPTR = (float) (*dividendPTR) / (* divisorPTR) ;
     ++divisorPTR; ++dividendPTR; ++outPTR;
  }
 return 1;
}


int multiply2DMatricesF(float * out , float * mult1 , float * mult2 , unsigned int width , unsigned int height , unsigned int channels)
{
  float *mult1PTR=mult1 , *mult2PTR=mult2 , *outPTR=outPTR , *outLimit=outPTR + (width * height * channels);

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

