#include "bilateralFilter.h"



int bilateralFilter(unsigned char * target,  unsigned int targetWidth , unsigned int targetHeight ,
                    unsigned char * source,  unsigned int sourceWidth , unsigned int sourceHeight ,

                    unsigned char * convolutionMatrix , unsigned int convolutionMatrixWidth , unsigned int convolutionMatrixHeight , unsigned int divisor ,

                    unsigned int tX,  unsigned int tY  ,
                    unsigned int sX,  unsigned int sY  ,
                    unsigned int patchWidth , unsigned int patchHeight
                   )
{
 return 0;
}


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
