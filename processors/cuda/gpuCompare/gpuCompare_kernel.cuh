/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/* Simple kernel computes a Stereo Disparity using CUDA SIMD SAD intrinsics. */

#ifndef _STEREODISPARITY_KERNEL_H_
#define _STEREODISPARITY_KERNEL_H_

#define blockSize_x 32
#define blockSize_y 8

// RAD is the radius of the region of support for the search
#define RAD 8
// STEPS is the number of loads we must perform to initialize the shared memory area
// (see convolution CUDA Sample for example)
#define STEPS 3

texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex2Dneedle;
texture<unsigned int, cudaTextureType2D, cudaReadModeElementType> tex2Dhaystack;


////////////////////////////////////////////////////////////////////////////////
// This function applies the video intrinsic operations to compute a
// sum of absolute differences.  The absolute differences are computed
// and the optional .add instruction is used to sum the lanes.
//
// For more information, see also the documents:
//  "Using_Inline_PTX_Assembly_In_CUDA.pdf"
// and also the PTX ISA documentation for the architecture in question, e.g.:
//  "ptx_isa_3.0K.pdf"
// included in the NVIDIA GPU Computing Toolkit
////////////////////////////////////////////////////////////////////////////////
__device__ unsigned int __usad4(unsigned int A, unsigned int B, unsigned int C=0)
{
    unsigned int result;
#if (__CUDA_ARCH__ >= 300) // Kepler (SM 3.x) supports a 4 vector SAD SIMD
    asm("vabsdiff4.u32.u32.u32.add" " %0, %1, %2, %3;": "=r"(result):"r"(A), "r"(B), "r"(C));
#else // SM 2.0            // Fermi  (SM 2.x) supports only 1 SAD SIMD, so there are 4 instructions
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b0, %2.b0, %3;": "=r"(result):"r"(A), "r"(B), "r"(C));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b1, %2.b1, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b2, %2.b2, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
    asm("vabsdiff.u32.u32.u32.add" " %0, %1.b3, %2.b3, %3;": "=r"(result):"r"(A), "r"(B), "r"(result));
#endif
    return result;
}

////////////////////////////////////////////////////////////////////////////////
//! Simple stereo disparity kernel to test atomic instructions
//! Algorithm Explanation:
//! For stereo disparity this performs a basic block matching scheme.
//! The sum of abs. diffs between and area of the candidate pixel in the left images
//! is computed against different horizontal shifts of areas from the right.
//! The shift at which the difference is minimum is taken as how far that pixel
//! moved between left/right image pairs.   The recovered motion is the disparity map
//! More motion indicates more parallax indicates a closer object.
//! @param g_img1  image 1 in global memory, RGBA, 4 bytes/pixel
//! @param g_img2  image 2 in global memory
//! @param g_odata disparity map output in global memory,  unsigned int output/pixel
//! @param w image width in pixels
//! @param h image height in pixels
////////////////////////////////////////////////////////////////////////////////
/*
__global__ void
compareImagesKernel(
                      unsigned int *g_img0, unsigned int *g_img1,
                      unsigned int *g_odata,
                      int w, int h
                      //,int minDisparity, int maxDisparity
                      )
{
int minDisparity=1;
int maxDisparity=3;
    // access thread id
    const int tidx = blockDim.x * blockIdx.x + threadIdx.x;
    const int tidy = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int sidx = threadIdx.x+RAD;
    const unsigned int sidy = threadIdx.y+RAD;

    unsigned int imLeft;
    unsigned int imRight;
    unsigned int cost;
    unsigned int bestCost = 9999999;
    unsigned int bestDisparity = 0;
    __shared__ unsigned int diff[blockSize_y+2*RAD][blockSize_x+2*RAD];

    // store needed values for left image into registers (constant indexed local vars)
    unsigned int imLeftA[STEPS];
    unsigned int imLeftB[STEPS];

    for (int i=0; i<STEPS; i++)
    {
        int offset = -RAD + i*RAD;
        imLeftA[i] = tex2D(tex2Dleft, tidx-RAD, tidy+offset);
        imLeftB[i] = tex2D(tex2Dleft, tidx-RAD+blockSize_x, tidy+offset);
    }

    // for a fixed camera system this could be hardcoded and loop unrolled
    //for (int d=minDisparity; d<=maxDisparity; d++)
    int d=0;
    {
        //LEFT
#pragma unroll
        for (int i=0; i<STEPS; i++)
        {
            int offset = -RAD + i*RAD;
            //imLeft = tex2D( tex2Dleft, tidx-RAD, tidy+offset );
            imLeft = imLeftA[i];
            imRight = tex2D(tex2Dright, tidx-RAD+d, tidy+offset);
            cost = __usad4(imLeft, imRight);
            diff[sidy+offset][sidx-RAD] = cost;
        }

        //RIGHT
#pragma unroll

        for (int i=0; i<STEPS; i++)
        {
            int offset = -RAD + i*RAD;

            if (threadIdx.x < 2*RAD)
            {
                //imLeft = tex2D( tex2Dleft, tidx-RAD+blockSize_x, tidy+offset );
                imLeft = imLeftB[i];
                imRight = tex2D(tex2Dright, tidx-RAD+blockSize_x+d, tidy+offset);
                cost = __usad4(imLeft, imRight);
                diff[sidy+offset][sidx-RAD+blockSize_x] = cost;
            }
        }

        __syncthreads();

        // sum cost horizontally
#pragma unroll

        for (int j=0; j<STEPS; j++)
        {
            int offset = -RAD + j*RAD;
            cost = 0;
#pragma unroll

            for (int i=-RAD; i<=RAD ; i++)
            {
                cost += diff[sidy+offset][sidx+i];
            }

            __syncthreads();
            diff[sidy+offset][sidx] = cost;
            __syncthreads();

        }

        // sum cost vertically
        cost = 0;
#pragma unroll

        for (int i=-RAD; i<=RAD ; i++)
        {
            cost += diff[sidy+i][sidx];
        }

        // see if it is better or not
        if (cost < bestCost)
        {
            bestCost = cost;
            bestDisparity = d+8;
        }

        __syncthreads();

    }

    if (tidy < h && tidx < w)
    {
        g_odata[tidy*w + tidx] = bestDisparity;
    }
}
*/

__global__ void
compareImagesKernel(
                      unsigned int *g_needle,
                      unsigned int needleWidth,
                      unsigned int needleHeight,

                      unsigned int *g_haystack,
                      unsigned int haystackWidth,
                      unsigned int haystackHeight,


                      unsigned int *g_odata,

                      unsigned int haystackTilesX,
                      unsigned int haystackTilesY
                      )
{
    // access thread id

    const unsigned int haystackPixelX = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int haystackPixelY = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int needlePixelX = haystackPixelX % needleWidth;
    const unsigned int needlePixelY = haystackPixelY % needleHeight;


    const unsigned int outputElement= haystackTilesX + (haystackTilesY*haystackTilesX);

    //This will be faster
    //__shared__ unsigned int diff[blockSize_y+2*RAD][blockSize_x+2*RAD];

    if ((haystackPixelY < haystackHeight) && (haystackPixelX+blockSize_x < haystackHeight))
    {
    #pragma unroll
     for (int i=0; i<blockSize_x; i++)
       {
        unsigned int valA = tex2D(tex2Dneedle, needlePixelX, needlePixelY+i);
        unsigned int valB = tex2D(tex2Dhaystack, haystackPixelX, haystackPixelY+i);
        g_odata[outputElement]= __usad(valA,valB,0);
       }
    }

   __syncthreads();

}






#define MEMPLACE4(x,y,width) ( ( y * ( width * 4 ) ) + (x*4) )

unsigned int sadRGBA(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
                     unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
                     unsigned int width , unsigned int height)
{
  unsigned int sum=0;
  unsigned int lineNum=0;

  unsigned int sX2 = sX+width;
  unsigned int sY2 = sY+height;

  unsigned int tX2 = tX+width;
  unsigned int tY2 = tY+height;


  unsigned char *  sourcePTR      = source+ MEMPLACE4(sX,sY,sourceWidth);
  unsigned char *  sourceLimitPTR = source+ MEMPLACE4(sX2,sY2,sourceWidth);
  unsigned char *  sourceFrameLimitPTR = source + sourceWidth*sourceHeight*4;
  unsigned int     sourceLineSkip = (sourceWidth-width) * 4;
  unsigned char *  sourceLineLimitPTR = sourcePTR + (width*4) -4; /*-3 is required here*/
  if (sourceLimitPTR>sourceFrameLimitPTR) { sourceFrameLimitPTR = sourceLimitPTR; }
/*
  fprintf(stderr,"SOURCE (RGB size %u/%u)  Starts at %u,%u and ends at %u,%u\n",sourceWidth,sourceHeight,sX,sY,sX2,sY2);
  fprintf(stderr,"sourcePTR is %p , limit is %p \n",sourcePTR,sourceLimitPTR);
  fprintf(stderr,"sourceLineSkip is %u\n",        sourceLineSkip);
  fprintf(stderr,"sourceLineLimitPTR is %p\n",sourceLineLimitPTR);*/


  unsigned char * targetPTR      = target + MEMPLACE4(tX,tY,targetWidth);
  unsigned char * targetLimitPTR = target + MEMPLACE4(tX2,tY2,targetWidth);
  unsigned char * targetFrameLimitPTR = target + targetWidth*targetHeight*4;
  if (targetLimitPTR>targetFrameLimitPTR) { targetFrameLimitPTR = targetLimitPTR; }

  unsigned int targetLineSkip = (targetWidth-width) * 4;
  unsigned char * targetLineLimitPTR = targetPTR + (width*4) -4; /*-3 is required here*/
  /*
  fprintf(stderr,"TARGET (RGB size %u/%u)  Starts at %u,%u and ends at %u,%u\n",targetWidth,targetHeight,tX,tY,tX2,tY2);
  fprintf(stderr,"targetPTR is %p , limit is %p \n",targetPTR,targetLimitPTR);
  fprintf(stderr,"targetLineSkip is %u\n", targetLineSkip);
  fprintf(stderr,"targetLineLimitPTR is %p\n",targetLineLimitPTR);*/

  while ( (sourcePTR < sourceLimitPTR) && ( targetPTR+4 < targetLimitPTR ) && (lineNum<height))
  {

     while ( (sourcePTR < sourceLineLimitPTR) && ((targetPTR+4 < targetLineLimitPTR)) )
     {
        sum+= abs( (int) *targetPTR - *sourcePTR ); ++targetPTR; ++sourcePTR;
        sum+= abs( (int) *targetPTR - *sourcePTR ); ++targetPTR; ++sourcePTR;
        sum+= abs( (int) *targetPTR - *sourcePTR ); ++targetPTR; ++sourcePTR;
        /*sum+= abs( (int) *targetPTR - *sourcePTR );*/ ++targetPTR; ++sourcePTR;
     }

    sourceLineLimitPTR += sourceWidth*4;
    targetLineLimitPTR += targetWidth*4;
    sourcePTR+=sourceLineSkip;
    targetPTR+=targetLineSkip;

    //fprintf(stderr,"Line %u , Score %u \n",lineNum,sum);
    ++lineNum;
  }

 return sum;
}







void compareImagesCPU(
                       unsigned char *needle,
                       unsigned int needleWidth,
                       unsigned int needleHeight,

                       unsigned char *haystack,
                       unsigned int haystackWidth,
                       unsigned int haystackHeight,

                       unsigned int haystackItemsX,
                       unsigned int haystackItemsY,

                       unsigned int *odata
                     )
{
  unsigned int scoreOutputID=0;
  unsigned int currentScore=0;


  for (int hY=0; hY<haystackItemsY; hY++)
  {
   for (int hX=0; hX<haystackItemsX; hX++)
    {
        if  ( (hY==haystackItemsY-1) && (hX==haystackItemsX-1) )
        {
          fprintf(stderr,"Edge case does not work.. \n");
          odata[scoreOutputID]=6666666;
        } else
        {
        fprintf(stderr,"Tile(%u,%u)",hX,hY);
         odata[scoreOutputID]=sadRGBA(
                                       haystack,
                                       hX*needleWidth,  hY*needleHeight , haystackWidth , haystackHeight ,
                                       needle  ,
                                       0 , 0 , needleWidth , needleHeight ,
                                       needleWidth , needleHeight
                                      );

        }

        ++scoreOutputID;
    }
  }
}
#endif // #ifndef _STEREODISPARITY_KERNEL_H_
