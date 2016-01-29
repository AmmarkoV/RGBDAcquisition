#include "imageMatrix.h"
#include <stdio.h>
#include <stdlib.h>

unsigned char* divideTwoImages1Ch(unsigned char *  divisor , unsigned char * divider , unsigned int width,unsigned int height )
{
  char *res = (char*) malloc(sizeof(char) * width * height );

  char *divisorPTR = divisor;
  char *dividerPTR = divider;
  char *resPTR = res;
  char *resLimit = res+width*height;
  if (res!=0)
   {
     while(resPTR<resLimit)
     {
       *resPTR = (unsigned char) *divisor / *divider;
       ++resPTR;
       ++divisor;
       ++divider;
     }
   }
  return res;
}

int divide2DMatricesF(float * out , float * divider , float * divisor , unsigned int width , unsigned int height )
{
  float *dividerPTR=divider , *divisorPTR=divisor , *outPTR=outPTR , *outLimit=outPTR + (width * height);

  while (outPTR < outLimit)
  {
     *outPTR = (float) (*dividerPTR) / (* divisorPTR) ;
     ++divisorPTR; ++dividerPTR; ++outPTR;
  }
 return 1;
}


int multiply2DMatricesF(float * out , float * mult1 , float * mult2 , unsigned int width , unsigned int height )
{
  float *mult1PTR=mult1 , *mult2PTR=mult2 , *outPTR=outPTR , *outLimit=outPTR + (width * height);

  while (outPTR < outLimit)
  {
     *outPTR = (float) (*mult1PTR) * (* mult2PTR) ;
     ++mult1PTR; ++mult2PTR; ++outPTR;
  }
 return 1;
}

int multiply2DMatricesFWithUC(float * out , float * mult1 , unsigned char * mult2 , unsigned int width , unsigned int height )
{
  float *mult1PTR=mult1 , *outPTR=outPTR , *outLimit=outPTR + (width * height);
  unsigned char *  mult2PTR=mult2;

  while (outPTR < outLimit)
  {
     *outPTR = (float) (*mult1PTR) * (* mult2PTR) ;
     ++mult1PTR; ++mult2PTR; ++outPTR;
  }
 return 1;
}

