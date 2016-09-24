#include "bilateralFilter.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define sqrt2 1.414213562

static inline int isDimensionOdd(unsigned int dimension)
{
 return ( dimension % 2 !=0 );
}

static inline float absFSub(float  value1,float  value2)
{
 if (value1>value2) { return (float ) value1-value2; }
 return (float ) value2-value1;
}

static inline int isNaN(float f)
{
  return ( f != f );
}

static inline float * getSpatialDifferenceMatrix(unsigned int dimension,float * divisor,float id)
{
  float * newMat = (float*) malloc(sizeof(float) * dimension * dimension );
  unsigned int x=0,y=0;

  *divisor=0;

  fprintf(stderr,"getSpatialDifferenceMatrix(%u) = \n",dimension);

  if(newMat!=0)
  {
   unsigned int centerElement = (unsigned int) dimension /2 ;
   float xMin,yMin;

   float * newMatPtr = newMat;
   for (y=0; y<dimension; y++)
   {
    for (x=0; x<dimension; x++)
    {
      xMin = ( (float) centerElement - x );
      yMin = ( (float) centerElement - y );

      *newMatPtr = (float) sqrt( (xMin*xMin) + (yMin*yMin) ) / id;
      *newMatPtr = exp( (*newMatPtr) * (*newMatPtr) * 0.5 );
      *divisor=*divisor + *newMatPtr;

      fprintf(stderr,"%0.4f    ",*newMatPtr);

      ++newMatPtr;
    }
    fprintf(stderr,"\n");
   }
  }

  return newMat;
}


static inline void doGenericBilateralFilterKernel (
                                             unsigned char * kernelStart,
                                             unsigned int dimension ,
                                             unsigned int sourceWidth ,
                                             float id, float cd ,
                                             float * spatialDifferences ,
                                             float spatialDifferenceDivisor ,
                                             float * tmpBuf ,
                                             unsigned char * output
                                            )
{
  // ================================================================
  unsigned int lineOffset = ( sourceWidth*3 ) - (dimension*3) ;
  unsigned char * kernelPTR = kernelStart;
  unsigned char * kernelLineEnd = kernelStart + (3*dimension);
  unsigned char * kernelEnd = kernelStart + (sourceWidth*3*dimension);
  float * tmpBufPTR = tmpBuf;
  //Get all input in tmpBuf so that we only access it..
  while (kernelPTR < kernelEnd)
  {
    while (kernelPTR < kernelLineEnd)
    {
     *tmpBufPTR = (float) *kernelPTR;
     ++tmpBufPTR;
     ++kernelPTR;
    }
   kernelPTR+=lineOffset;
   kernelLineEnd+=( sourceWidth*3 );
  }
  // ================================================================



  unsigned int dimensionOffsetForCenter = (unsigned int) (dimension*dimension)/2;
  dimensionOffsetForCenter = dimensionOffsetForCenter * 3;
  float * centerElementR = tmpBuf+dimensionOffsetForCenter+0;
  float * centerElementG = tmpBuf+dimensionOffsetForCenter+1;
  float * centerElementB = tmpBuf+dimensionOffsetForCenter+2;
  float * kernelFPTR = tmpBuf;
  float * kernelFPTREnd = tmpBuf + (dimension*dimension*3);

  float outputF[3]={0};
  float resR,resG,resB;


  float * spatialDifferencesPTR = spatialDifferences;
  float sumWeight=0.0;

  while (kernelFPTR<kernelFPTREnd)
  {
    resR = (float) (*kernelFPTR+0) - (*centerElementR);
    resG = (float) (*kernelFPTR+1) - (*centerElementG);
    resB = (float) (*kernelFPTR+2) - (*centerElementB);

    float colorDist = (float) sqrt( (float)( (resR*resR) + (resG*resG) + (resB*resB) ) );
    float coW = (float) colorDist/cd;

    float exp_imW_Mul_imW_Mul_0_5 = *spatialDifferencesPTR;   ++spatialDifferencesPTR;//table is precalculated
    float dividend = ( exp_imW_Mul_imW_Mul_0_5 * exp(coW*coW*0.5) );
    float currWeight = (float) 1.0f/dividend;

     if ( !isNaN(currWeight)  )
    {
     //fprintf(stderr,"currWeight ( 1/%0.2f = %0.2f )",dividend,currWeight);
     sumWeight += currWeight;

     outputF[0] += (float) currWeight * (*kernelFPTR); ++kernelFPTR;
     outputF[1] += (float) currWeight * (*kernelFPTR); ++kernelFPTR;
     outputF[2] += (float) currWeight * (*kernelFPTR); ++kernelFPTR;
    }
     else
    {
      kernelFPTR+=3;
    }

  }
  //fprintf(stderr,"out ( %0.2f %0.2f %0.2f ) ",outputF[0],outputF[1],outputF[2]);

  if (sumWeight==0)
  {
   outputF[0] =  (*centerElementR);
   outputF[1] =  (*centerElementG);
   outputF[2] =  (*centerElementB);
  } else
  {
   outputF[0] /= sumWeight;
   outputF[1] /= sumWeight;
   outputF[2] /= sumWeight;
  }

  output[0] = (unsigned char) outputF[0];
  output[1] = (unsigned char) outputF[1];
  output[2] = (unsigned char) outputF[2];


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

  if(id==0)
  {
    fprintf(stderr,"Not accepting zero spatial weight..\n");
    return 0;
  }


  if (cd==0)
  {
    fprintf(stderr,"Not accepting zero color weight..\n");
    return 0;
  }


  if (!isDimensionOdd(dimension))
  {
    fprintf(stderr,"Not accepting even dimensions , there is no central point..\n");
    return 0;
  }

 unsigned int kernelWidth=dimension,kernelHeight=dimension;



  float * tmpMat = (float*) malloc(sizeof(float) * dimension * dimension *3 );
  if (tmpMat==0)
  {
   fprintf(stderr,"Could not allocate a tmp matrix..\n");
    return 0;
  }

  float spatialDifferenceDivisor=0;
  float * spatialDifferences = getSpatialDifferenceMatrix(dimension,&spatialDifferenceDivisor,id);

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
      doGenericBilateralFilterKernel( sourceScanlinePTR , dimension , sourceWidth ,  id, cd  , spatialDifferences , spatialDifferenceDivisor , tmpMat , outputRGB );

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


 free(spatialDifferences);
 free(tmpMat);
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
