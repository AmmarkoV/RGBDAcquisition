#include "bilateralFilter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define sqrt2 1.414213562

inline int isDimensionOdd(unsigned int dimension)
{
 return ( dimension % 2 !=0 );
}


inline unsigned char absSub(unsigned char value1,unsigned char value2)
{
 if (value1>value2) { return (unsigned char) value1-value2; }
 return (unsigned char) value2-value1;
}


inline unsigned char scaleIntensity(unsigned char sideValue,unsigned char centerValue,unsigned char absValueDiff,float distance)
{
  float scaledValueDiff =  (float) absValueDiff / distance;
  float sideValueScaled = 0.0 ;


  if (centerValue>sideValue) { sideValueScaled = centerValue - scaledValueDiff; } else
                             { sideValueScaled = centerValue + scaledValueDiff; }


  scaledValueDiff = (float) (sideValueScaled + centerValue) / 2;

  unsigned char resultUC = (unsigned char) scaledValueDiff;

  return resultUC;
}



inline float * getSpatialDifferenceMatrix(unsigned int dimension,float * divisor)
{
  float * newMat = (float*) malloc(sizeof(float) * dimension * dimension );
  unsigned int x=0,y=0;

  *divisor=0;

  if(newMat==0)
  {
   unsigned int centerElement = (unsigned int) dimension /2 ;
   float xMin,yMin;

   float * newMatPtr = newMat;
   for (y=0; y<dimension; y++)
   {
    for (x=0; x<dimension; x++)
    {
      xMin = ( centerElement - x );
      yMin = ( centerElement - y );

      *newMatPtr = (float) sqrt( (xMin*xMin) + (yMin*yMin) );
      *divisor=*divisor + *newMatPtr;
       ++newMatPtr;
    }
   }
  }

  return newMat;
}


inline void do3x3BilateralFilterKernel( unsigned char * kernelStart, unsigned int sourceWidth , float id, float cd , float * spatialDifferences , float spatialDifferenceDivisor , unsigned char * output)
{
  /* 0a 1b 2c
     3d 4e 5f   e is anchor point (kernel center)
     6g 7h 8ii */

  unsigned int channel=0;
  unsigned char *a , *b , *c , *d , *e , *f ,*g, *h , *ii;
  a = kernelStart;         b = a+3;   c = b+3;
  d = a+sourceWidth * 3;   e = d+3;   f = e+3;
  g = d+sourceWidth * 3;   h = g+3;   ii = h+3;

 for (channel=0; channel<3; channel++)
 {

  unsigned char intensityDifferences[3*3];
  intensityDifferences[0]= absSub(*e,*a); intensityDifferences[1]= absSub(*e,*b); intensityDifferences[2]= absSub(*e,*c);
  intensityDifferences[3]= absSub(*e,*d); intensityDifferences[4]= 0;             intensityDifferences[5]= absSub(*e,*f);
  intensityDifferences[6]= absSub(*e,*g); intensityDifferences[7]= absSub(*e,*h); intensityDifferences[8]= absSub(*e,*ii);

  unsigned char intensityAbsolute[3*3];
  intensityAbsolute[0]= *a; intensityAbsolute[1]= *b; intensityAbsolute[2]= *c;
  intensityAbsolute[3]= *d; intensityAbsolute[4]= *e; intensityAbsolute[5]= *f;
  intensityAbsolute[6]= *g; intensityAbsolute[7]= *h; intensityAbsolute[8]= *ii;

  unsigned int i=0,intensitySum=0,intensityDivisor=0;

  intensitySum=*e;
  intensityDivisor=1;

  for (i=0; i<9; i++)
  {
    if ( intensityDifferences[i] > cd )
    {
      //intensitySum+=intensityAbsolute[i];
      intensitySum+=scaleIntensity(intensityAbsolute[i],*e,intensityDifferences[i],spatialDifferences[i]);
      ++intensityDivisor;
    }
  }

  float resultValueF=0.0;
  if (intensityDivisor!=0)
  {
   resultValueF = (float) intensitySum / intensityDivisor;
  }

  unsigned char resultValue=(unsigned char) resultValueF;
  if (channel==0) {   output[0] = resultValue; }
  if (channel==1) {   output[1] = resultValue; }
  if (channel==2) {   output[2] = resultValue; }

   //Go to the next channel
   ++a; ++b; ++c; ++d; ++e; ++f; ++g; ++h; ++ii;
 }





 return;
}


int bilateralFilterInternal(unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                    unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,

                    float id, float cd , unsigned int dimension
                   )
{
  unsigned char * sourcePTR = source;
  unsigned char * sourceLimit = source+(sourceWidth*sourceHeight*3) ;
  unsigned char * targetPTR = target;
  unsigned char * targetLimit = target+(targetWidth*targetHeight*3) ;


  if (!isDimensionOdd(dimension))
  {
    fprintf(stderr,"Not accepting even dimensions , there is no central point..\n");
    return 0;
  }
  if ( dimension != 3 )
  {
    fprintf(stderr,"Cannot perform bilateral filter for dimensions other than 3x3\n");
    return 0;
  }
 unsigned int kernelWidth=dimension,kernelHeight=dimension;
 unsigned int workableAreaStartX=0,workableAreaEndX=sourceWidth-kernelWidth,workableAreaStartY=sourceHeight-kernelHeight,workableAreaEndY;

  float spatialDifferenceDivisor=0;
  float * spatialDifferences = getSpatialDifferenceMatrix(dimension,&spatialDifferenceDivisor);

  if (spatialDifferences==0)
  {
   fprintf(stderr,"Could not allocate a spatial matrix..\n");
    return 0;
  }


  unsigned int x=0,y=0;

  while (sourcePTR < sourceLimit)
  {
   unsigned char * sourceScanlinePTR = sourcePTR;
   unsigned char * sourceScanlineEnd = sourcePTR+(sourceWidth*3);

   unsigned char * targetScanlinePTR = targetPTR;
   unsigned char * targetScanlineEnd = targetPTR+(targetWidth*3);

   if (x+kernelWidth>=sourceWidth)
   {

   } else
   if (y+kernelHeight>=sourceHeight)
   {
      //We are on the right side of our buffer
   } else
   {
   //Get all the valid configurations of the scanline
   while (sourceScanlinePTR < sourceScanlineEnd)
    {
      unsigned char outputRGB[3];
      do3x3BilateralFilterKernel( sourceScanlinePTR , sourceWidth ,  id, cd  , spatialDifferences , spatialDifferenceDivisor , outputRGB );

      unsigned char * outputR = targetScanlinePTR + (targetWidth*3) + 3;
      unsigned char * outputG = outputR+1;
      unsigned char * outputB = outputG+1;

      *outputR = outputRGB[0];
      *outputG = outputRGB[1];
      *outputB = outputRGB[2];

      ++x;
      sourceScanlinePTR+=3;
      targetScanlinePTR+=3;
    }
     sourcePTR = sourceScanlineEnd-3;
     targetPTR = targetScanlineEnd-3;

   }

   //Keep X,Y Relative positiions
   ++x;
   if (x>=sourceWidth) { x=0; ++y; }
   sourcePTR+=3;
   targetPTR+=3;
  }

 fprintf(stderr,"\n",x,y);

 free(spatialDifferences);
 return 1;
}




int bilateralFilter(unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                    unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,

                    float id, float cd , unsigned int dimension
                   )
{
 fprintf(stderr,"bilateralFilter(%0.2f %0.2f %u )\n",id,cd,dimension);
 return bilateralFilterInternal(target,targetWidth,targetHeight,source,sourceWidth,sourceHeight,id,cd,dimension);
}

/*
void convolution(unsigned char *_in, unsigned char *_out, int width, int height, int halfkernelsize, float id, float cd)
{
  int kernelDim = 2*halfkernelsize+1;


  float sumWeight = 0;

  int x,y,i,j;
  unsigned int ctrIdx = y*width + x;
  unsigned int _sum[3]={0};

  float ctrPix[3];
  ctrPix[0] = _in[ctrIdx+0];
  ctrPix[1] = _in[ctrIdx+1];
  ctrPix[2] = _in[ctrIdx+2];

  // neighborhood of current pixel
  int kernelStartX, kernelEndX, kernelStartY, kernelEndY;
  kernelStartX = x-halfkernelsize;
  kernelEndX   = x+halfkernelsize;
  kernelStartY = y-halfkernelsize;
  kernelEndY   = y+halfkernelsize;

  for(j= kernelStartY; j<= kernelEndY; j++)
    {
      for(i= kernelStartX; i<= kernelEndX; i++)
    {
      unsigned int idx = max(0, min(j, height-1))*width + max(0, min(i,width-1));

      float curPix[3];
      curPix[0] = _in[idx+0];
      curPix[1] = _in[idx+1];
      curPix[2] = _in[idx+2];


      float currWeight;

      // define bilateral filter kernel weights
      float imageDist = sqrt( (float)((i-x)*(i-x) + (j-y)*(j-y)) );

      float colorDist = sqrt( (float)( (curPix[0] - ctrPix[0])*(curPix[0] - ctrPix[0]) +
                       (curPix[1] - ctrPix[1])*(curPix[1] - ctrPix[1]) +
                       (curPix[2] - ctrPix[2])*(curPix[2] - ctrPix[2]) ) );

      currWeight = 1.0f/(exp((imageDist/id)*(imageDist/id)*0.5)*exp((colorDist/cd)*(colorDist/cd)*0.5));
      sumWeight += currWeight;

      _sum[0] += currWeight*curPix[0];
      _sum[1] += currWeight*curPix[1];
      _sum[2] += currWeight*curPix[2];
    }
    }

  _sum[0] /= sumWeight;
  _sum[1] /= sumWeight;
  _sum[2] /= sumWeight;

  _out[ctrIdx+0] = (int)(floor(_sum[0]));
  _out[ctrIdx+1] = (int)(floor(_sum[1]));
  _out[ctrIdx+2] = (int)(floor(_sum[2]));
  //_out[ctrIdx] = _in[ctrIdx].w;
}
*/
