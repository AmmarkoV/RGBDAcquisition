#include "dericheRecursiveGaussian.h"
#include "constantTimeBilateralFiltering.h"
#include <math.h>


struct ctbfPool
{
  float * Wk;
  float * Jk;
  unsigned char * JBk;
};


inline unsigned char* divideTwoImages1Ch(unsigned char *  divisor , unsigned char * divider , unsigned int width,unsigned int height )
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



#define SR  0.006
#define SR2 2*SR
inline float wKResponse(float kBin , float in , float divider )
{
  float response = -1 * ( in - kBin )  * ( in - kBin );
  response = response / (SR2);
  return (float) exp(response)/divider;
}

int constantTimeBilateralFilter(
                                unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,
                                unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                                float sigma ,
                                unsigned int bins
                               )
{
  unsigned int i=0;
  struct ctbfPool * ctfbp = (struct ctbfPool *) malloc( sizeof(struct ctbfPool) * bins );
  if (ctfbp == 0 ) { return 0; }

  for (i=0; i<bins; i++)
  {
    ctfbp[i].Wk =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight );
    ctfbp[i].Jk =  ( float * ) malloc( sizeof( float ) * sourceWidth * sourceHeight );
  }

  float maxVal=255;
  float step = maxVal / ( bins-1);
  float divider = sqrt( M_PI *(SR2));


for (i=0; i<bins; i++)
{

 //Todo
 // wKResponse(float kBin , float in , float divider )


 /*
 dericheRecursiveGaussianGray(
                              ctfbp[i].Jk,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int channels,
                              TMP1OUTPUT,  unsigned int targetWidth , unsigned int targetHeight ,
                              sigma , 0
                             );


 dericheRecursiveGaussianGray(
                              ctfbp[i].Wk,  unsigned int sourceWidth , unsigned int sourceHeight , unsigned int channels,
                              TMP2OUTPUT,  unsigned int targetWidth , unsigned int targetHeight ,
                              sigma , 0
                             );

  ctfbp[i].JBk divideTwoImages1Ch(TMP1OUTPUT,TMP2OUTPUT,sourceWidth,sourceHeight);

  */
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


}
