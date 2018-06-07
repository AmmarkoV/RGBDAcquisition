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
reduce2(unsigned int *g_idata, unsigned int *g_odata, unsigned int n)
{
    __shared__ unsigned int sdata[64*64];

    // load shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? g_idata[i] : 0;

    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }

        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__host__
void reduceThe4ResultsPerTileIntoOneInPlace(unsigned int * results , unsigned int numberOfTilesX,unsigned int numberOfTilesY,unsigned int resultsPerTile)
{
  unsigned int outputCounter =0 ;
  unsigned int inputCounter  =0 ;

  unsigned int i=0;
  for (i=0; i<numberOfTilesX*numberOfTilesY; i++)
  {
    unsigned int pos = inputCounter;
    unsigned int currentResult = results[pos+0] + results[pos+1] + results[pos+2] + results[pos+3];

    results[outputCounter]=currentResult;

    outputCounter+=1;
    inputCounter+=4;
  }

  for (i=outputCounter; i<numberOfTilesX*numberOfTilesY*4; i++)
  {
    results[i]=0;
  }
}




__global__ void
compareImagesKernel(
                      //This is a global memory buffer to hold the intermediate results..
                      //unsigned int *g_subtracted,

                      unsigned int *g_needle,
                      unsigned int needleWidth,   unsigned int needleHeight,

                      unsigned int *g_haystack,
                      unsigned int haystackWidth, unsigned int haystackHeight,

                      unsigned int *g_odata ,
                      unsigned int *g_bothData ,
                      unsigned int *g_missData ,

                      unsigned int maximumDifference
                      )
{
    /*
              We have two rectangular regions (textures), The one is our BIG haystack ( 1024x1024 pixel )  and the second is our small ( 64x64 pixel ) needle

                    Haystack 1024x1024 pixels (16x16 tiles)                           Needle 64x64 pixels
               ________________________________________________                              __
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|                             |__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|
              |__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|__|

              We basically want to compare our 64x64 needle to all of the boxes of our haystack in parallel..
              Since CUDA has tighter limitations though we cannot just tile on the 64x64 needle but we have to split
              it in smaller chunks in order to accomodate our GPU limits.


              The maximum threads possible per block is 1024 ( = 32x32 ) so using this configuration  we will split each 64x64 tile into 4  32x32 chunks.
              Each of the CUDA blocks will process 1/4 of a tile
              Each of the CUDA blocks will use 32x32 threads
              Each of the threads will process ONE pixel and we will maximize throughput that way..

                Zoomed:

              One Haystack Tile contains 64x64 pixels
              and will be computed by 4 different blocks
                ______________ ______________
               |              |              |
               |              |              |
               |  Block(X,Y)  | Block(X+1,Y) |
               |              |              |
               |______________|______________|
               |              |              |
               |              |              |
               | Block(X,Y+1) |Block(X+1,Y+1)|
               |              |              |
               |______________|______________|


               Doing the Substraction of Absolute Differences ( SAD ) is very straight-forward, however we want to also reduce
               the results to fewer values that can be then quickly transported back to system memory holding our final results
               We will output 4 results for each of the tiles and the CPU will have to do the final 4 part summation, this is faster
               than doing more global memory hits inside the CUDA code..

                 Zoomed:

                  Block (X,Y)
                ________________
               |                |
               |    Threads     |
               |    (32,32)     |
               | One for every  |
               |     pixel      |
               |________________|


              Each of the blocks will also have some shared memory ( sdata[32x32] ) which will hold its results



              Our final result vector will be formatted like this example :

              g_odata [ 16 x 16 x 4 ]
              --
              g_odata [ 0 ] = Result 1/4 of Tile 0,0
              g_odata [ 1 ] = Result 2/4 of Tile 0,0
              g_odata [ 2 ] = Result 3/4 of Tile 0,0
              g_odata [ 3 ] = Result 4/4 of Tile 0,0
              --
              g_odata [ 4 ] = Result 1/4 of Tile 1,0
              g_odata [ 5 ] = Result 2/4 of Tile 1,0
              g_odata [ 6 ] = Result 3/4 of Tile 1,0
              g_odata [ 7 ] = Result 4/4 of Tile 1,0
              --

              ...

              etc

              ...

              So in order for the CPU to get the final value for Tile 0 it will have to sum all of its parts..
              we can do that using the reduceThe4ResultsPerTileIntoOneInPlace call implemented above..

    */



    //Our big haystack image sized 1024x1024 ( 16x16 tiles sized 64x64 pixels each ) has arrived on the GPU, the particular
    //Each block will target a specific pixel of the haystack texture and here we find out which..
    const unsigned int haystackPixelX = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int haystackPixelY = blockDim.y * blockIdx.y + threadIdx.y;


    //Our needle image sized 64x64 is also a GPU texture
    //Each haystack pixel has to be compared with the according needle pixel (needlePixelX,needlePixelY)
    const unsigned int needlePixelX = haystackPixelX % needleWidth;
    const unsigned int needlePixelY = haystackPixelY % needleHeight;


    //We would also like to know which tile we are performing SAD for
    unsigned int haystackTileX = haystackPixelX / needleWidth;
    unsigned int haystackTileY = haystackPixelY / needleHeight;

    //Finally we calculate the number of haystack Tiles (16x16)
    unsigned int totalNumberOfHaystackTilesX  = haystackWidth / needleWidth;
  //unsigned int totalNumberOfHaystackTilesY  = haystackHeight / needleHeight; //<- not used
    //------------------------------------------------------------------------------------------


    //------------------------------------------------------------------------------------------
    //As mentioned in the intro comment for 1024x1024 we execute using 32x32 cuda blocks
    //Each block has a shared memory of threadSizeX x threadSizeY ( 32x32 )
    //Each of our 16x16 tiles sized (64x64) holds will be reflected by 4 blocks of sdata..!
    __shared__ unsigned int  sdata[threadSize_x*threadSize_y]; // 16384 of 49152 remaining bytes
    __shared__ unsigned char both[threadSize_x*threadSize_y];  //  4096 of 32768 remaining bytes
    __shared__ unsigned char miss[threadSize_x*threadSize_y];  //  4096 of 28672 remaining bytes
    //So each thread will use its sdata slot to keep one value
    unsigned int sdataPTR   = threadIdx.x + ( threadIdx.y * threadSize_x);
    //We also need the limit to make sure no overflows occur
    unsigned int sdataLimit = threadSize_x*threadSize_y;


    //If we are inside the compareable area we need to calulate the difference
    if ((haystackPixelY < haystackHeight) && (haystackPixelX < haystackWidth))
    {
        //We should get the needle and haystack values
        unsigned int needleValue   = tex2D(tex2Dneedle, needlePixelX, needlePixelY);
        unsigned int haystackValue = tex2D(tex2Dhaystack, haystackPixelX, haystackPixelY);

        //Calculate their absolute distance  | needleValue - haystackValue | + 0
        if (sdataPTR < sdataLimit)
        {
           sdata[sdataPTR] = min ( __usad(needleValue,haystackValue,0) , maximumDifference );
           both[sdataPTR]  = ((needleValue) && (haystackValue));
           miss[sdataPTR]  = (((needleValue) && (!haystackValue)) || ((!needleValue) && (haystackValue)) );
        }
    }

    //We need to make sure all threads are in sync to go on
    __syncthreads();


    //---------------------------------------------------------------------------
    //sdata now holds all results , we now need to sum them performing reduction
    //---------------------------------------------------------------------------

    // do horizontal reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1)
    {
        if (threadIdx.x < s)
        {
            sdata[sdataPTR] += sdata[sdataPTR + s];
            both[sdataPTR]  +=  both[sdataPTR + s];
            miss[sdataPTR]  +=  miss[sdataPTR + s];
        }
        __syncthreads();
    }

     __syncthreads();

    // do vertical reduction in shared mem
    for (unsigned int s=blockDim.y/2; s>0; s>>=1)
    {
        if (threadIdx.y < s)
        {
            sdata[sdataPTR] += sdata[sdataPTR + s*threadSize_x];
            both[sdataPTR]  +=  both[sdataPTR + s*threadSize_x];
            miss[sdataPTR]  +=  miss[sdataPTR + s*threadSize_x];
        }
        __syncthreads();
    }

    //Each tile has 4 chunks ..
    const unsigned int outputTileNumberOfChunksX  =  blockDim.x/ ( haystackWidth/needleWidth ) ;
    const unsigned int outputTileNumberOfChunksY  =  blockDim.y/ ( haystackHeight/needleHeight );

    const unsigned int chunkX    =  blockIdx.x%outputTileNumberOfChunksX;
    const unsigned int chunkY    =  blockIdx.y%outputTileNumberOfChunksY;
    const unsigned int chunkPTR  =  chunkX + (chunkY * outputTileNumberOfChunksX);


    //Finally we will output our sum here..
    const unsigned int outputTile         = haystackTileX*4  + (haystackTileY * totalNumberOfHaystackTilesX*4 );
    // write result for this block to global mem



    //Only one thread from each block is going to get inside the next statement
    if (
          (threadIdx.x == 0) &&
          (threadIdx.y == 0)
        )
    {
      //And is going to store its 1/4 of a result in the correct place
      g_odata[outputTile+chunkPTR] = sdata[0];
      g_bothData [outputTile+chunkPTR] = both[0];
      g_missData [outputTile+chunkPTR] = miss[0];
    }

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
