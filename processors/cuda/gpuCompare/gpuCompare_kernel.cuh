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

#define threadSize_x 32
#define threadSize_y 32


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


__global__ void
compareImagesKernel(
                      unsigned int *g_needle,
                      unsigned int needleWidth,   unsigned int needleHeight,

                      unsigned int *g_haystack,
                      unsigned int haystackWidth, unsigned int haystackHeight,

                      unsigned int *g_odata ,

                      unsigned int maximumDifference
                      )
{
    //Our big haystack image sized 1024x1024 ( 16x16 tiles ) has arrived on the GPU, the particular
    //Call is specifically targeted on haystack pixel (haystackPixelX,haystackPixelY)
    const unsigned int haystackPixelX = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int haystackPixelY = blockDim.y * blockIdx.y + threadIdx.y;

    //This is the result offset to temporarilly store all results
    //const unsigned int resultOffset = haystackPixelX + haystackPixelY * blockDim.x * gridDim.x;

    //Our needle image sized 64x64 is also on the GPU
    //Each haystack pixel has to be compared with the according needle pixel (needlePixelX,needlePixelY)
    //We can calculate this very easily
    const unsigned int needlePixelX = haystackPixelX % needleWidth;
    const unsigned int needlePixelY = haystackPixelY % needleHeight;

    //We would also like to know which tile we are performing SAD for
    unsigned int haystackTileX = haystackPixelX / needleWidth;
    unsigned int haystackTileY = haystackPixelY / needleHeight;

    unsigned int totalNumberOfHaystackTilesX  = haystackWidth / needleWidth;
    unsigned int totalNumberOfHaystackTilesY  = haystackHeight / needleHeight;

    //Finally we will output our sum here..
    const unsigned int outputElement= haystackTileX + (haystackTileY * totalNumberOfHaystackTilesX );

    //This will be faster
    //__shared__ unsigned int dest[16][16];


    //This will hold the difference
    unsigned int currentDifference;
    unsigned int bothHits=0;
    unsigned int oneHits=0;

    //We make sure all of our blocks have cleaned shared memory
     //dest[haystackTileX][haystackTileY]=0;
     __syncthreads();

    //Everything is in SYNC so now only if we are inside the compareable area
    if ((haystackPixelY < haystackHeight) && (haystackPixelX < haystackWidth))
    {
        //We should get the needle and haystack values
        unsigned int needleValue   = tex2D(tex2Dneedle, needlePixelX, needlePixelY);
        unsigned int haystackValue = tex2D(tex2Dhaystack, haystackPixelX, haystackPixelY);

        //Calculate their absolute distance  | needleValue - haystackValue | + 0
        currentDifference= __usad(needleValue,haystackValue,0); //__usad4

        bothHits =  ( (needleValue) && (haystackValue)  );
        //oneHits =   ( (needleValue) ^ (haystackValue)  );

        //If their absolute difference is more than a threshold , then threshold it
        //currentDifference = max( currentDifference , maximumDifference);

        //if (currentDifference>maximumDifference) { currentDifference=maximumDifference; }
    }

     //This is the group value that is faster to write to..!
/*
     if (
          (haystackTileX<totalNumberOfHaystackTilesX) &&
          (haystackTileY<totalNumberOfHaystackTilesY)
         )
         {
          dest[haystackTileX][haystackTileY]+=currentDifference;
         }
     __syncthreads();*/

     //This is probably wrong..!
          if (
          (haystackTileX<totalNumberOfHaystackTilesX) &&
          (haystackTileY<totalNumberOfHaystackTilesY)
         )
         {
          g_odata[outputElement] += currentDifference;// dest[haystackTileX][haystackTileY];
         }

     //Is this needed?
     //__syncthreads();
}




















/*
   ---------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------

   ---------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------
*/











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
