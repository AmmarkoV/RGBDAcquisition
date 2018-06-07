/*
Based on the NVIDA Sample
 *
 */

/* A CUDA program that demonstrates how to compute a stereo disparity map using
 *   SIMD SAD (Sum of Absolute Difference) intrinsics
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, kernels
#include <cuda_runtime.h>
#include "sdkHelper.h"
#include "gpuCompare_kernel.cuh"


// includes, project
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing


unsigned long tickBase = 0;
unsigned long GetTickCountMilliseconds()
{
   //This returns a monotnic "uptime" value in milliseconds , it behaves like windows GetTickCount() but its not the same..
   struct timespec ts;
   if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0) { return 0; }

   if (tickBase==0)
   {
     tickBase = ts.tv_sec*1000 + ts.tv_nsec/1000000;
     return 0;
   }

   return ( ts.tv_sec*1000 + ts.tv_nsec/1000000 ) - tickBase;
}



int iDivUp(int a, int b)
{
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}


////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

int writeResult(
                const char * device,
                const char * file1,
                const char * file2,
                unsigned int * odata,
                unsigned int w ,
                unsigned int h
               )
{
    unsigned int checkSum = 0;
    for (unsigned int i=0 ; i<w *h ; i++)
    {
        checkSum += odata[i];
    }
    printf("%s Checksum = %u, ",device, checkSum);

    // write out the resulting disparity image.
    unsigned char *dispOut = (unsigned char *)malloc(w*h);
    char fnameOut[512];
    snprintf(fnameOut,512,"output_%s_%s_%s.pgm",device,file1,file2);
    for (unsigned int i=0; i<w*h; i++) { dispOut[i] = (int)odata[i]; }
    for (unsigned int i=0; i<w*h; i++) { dispOut[i] = (int)odata[i]; }

    printf("%s image: <%s>\n", device , fnameOut);
    sdkSavePGM(fnameOut, dispOut, w, h);

    free(dispOut);
    return 1;
}


/*
int writeCUDALayoutToSVG(const char * svgOut,
                         unsigned int haystackWidth,unsigned int haystackHeight,
                         unsigned int needleWidth,unsigned int needleHeight,

                         unsigned int blocksX,unsigned int blocksY,
                         unsigned int threadsX,unsigned int threadsY
                         )
{
    FILE * fp=0;

    fp = fopen(svgOut,"w");

    unsigned int offX=100,offY=100,x=0,y=0;

    unsigned int tilesX=haystackWidth/needleWidth;
    unsigned int tilesY=haystackHeight/needleHeight;

    if (fp!=0)
    {
      fprintf(fp,"<svg width=\"1500\" height=\"3000\" >\n");
      fprintf(fp,"<rect width=\"1500\" height=\"3000\" style=\"fill:rgb(255,255,255);stroke-width:3;stroke:rgb(255,255,255)\" />\n");

      fprintf(fp,"<style>\n");
      fprintf(fp,"  .small { font: italic 13px sans-serif; }\n");
      fprintf(fp,"  .label { font: bold 30px sans-serif; }\n");
      fprintf(fp,"</style>\n");

      //fprintf(fp," <rect x=\"%u\" y=\"%u\"  width=\"%u\" height=\"%u\" style=\"fill:rgb(255,255,255);stroke-width:3;stroke:rgb(0,0,0)\" /> \n", offX,offY, haystackWidth,haystackHeight );

      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\">Needle</text>\n",1100+offX,offY-30);
      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\">Width %u</text>\n",1100+offX,offY+30,needleWidth);
      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\" transform=\"rotate(90,%u %u)\" >Height %u</text>\n",1150+offX-80,offY+20,1150+offX-80,offY+40,needleHeight);

             fprintf(fp," <rect x=\"%u\" y=\"%u\"  width=\"%u\" height=\"%u\" style=\"fill:rgb(123,123,123);stroke-width:3;stroke:rgb(50,0,0)\" /> \n",
                     1200+offX + x*needleWidth ,
                     offY + 80 + y*needleHeight,
                     needleWidth,
                     needleHeight );







      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\"> Haystack Width is %u pixels</text>\n",offX,offY-30,haystackWidth);
      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\" transform=\"rotate(90,%u %u)\" > Haystack Height is %u pixels</text>\n",offX-80,offY+20,offX-80,offY+20,haystackHeight);
        for (y=0; y<tilesY; y++)
         {
          for (x=0; x<tilesX; x++)
           {
             fprintf(fp," <rect x=\"%u\" y=\"%u\"  width=\"%u\" height=\"%u\" style=\"fill:rgb(123,123,123);stroke-width:3;stroke:rgb(50,0,0)\" /> \n",
                     offX + x*needleWidth ,
                     offY + y*needleHeight,
                     needleWidth,
                     needleHeight );

           }
         }


       offY+=1200;

      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\"> %u Blocks at Dimension X </text>\n",offX,offY-30,blocksX);
      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\" transform=\"rotate(90,%u %u)\" > %u Blocks at Dimension Y </text>\n",offX-80,offY+20,offX-80,offY+20,blocksY);
        //fprintf(fp," <rect x=\"%u\" y=\"%u\"  width=\"%u\" height=\"%u\" style=\"fill:rgb(255,255,255);stroke-width:3;stroke:rgb(0,0,0)\" /> \n", offX,offY, haystackWidth,haystackHeight );
        for (y=0; y<blocksX; y++)
         {
          for (x=0; x<blocksY; x++)
           {
             fprintf(fp," <rect x=\"%u\" y=\"%u\"  width=\"%u\" height=\"%u\" style=\"fill:rgb(123,123,123);stroke-width:3;stroke:rgb(50,0,0)\" /> \n",
                     offX + x*threadsX ,
                     offY + y*threadsY,
                     needleWidth,
                     needleHeight );

           }
         }
        for (y=0; y<tilesY; y++)
         {
          for (x=0; x<tilesX; x++)
           {
             fprintf(fp," <rect x=\"%u\" y=\"%u\"  width=\"%u\" height=\"%u\" style=\"fill:rgba(123,123,123,200);stroke-width:3;stroke:rgb(150,0,0)\" fill-opacity=\"0.4\"  /> \n",
                     offX + x*needleWidth ,
                     offY + y*needleHeight,
                     needleWidth,
                     needleHeight );

           }
         }

      offY+=1200;
      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\"> Thread Size ( %u , %u ) </text>\n",offX,offY-30,threadsX,threadsY);


      offY+=100;
      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\"> Each comparison is  performed</text>\n",offX,offY-30 );


      offY+=60;
      unsigned int blocksPerTileX = (unsigned int) needleWidth/threadsX;
      unsigned int blocksPerTileY = (unsigned int) needleHeight/threadsY;
      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\">by %u blocks (%u,%u) </text>\n",offX,offY-30,
              blocksPerTileX+blocksPerTileY,blocksPerTileX,blocksPerTileY
              );

      offY+=100;
      fprintf(fp," <text x=\"%u\" y=\"%u\" fill=\"red\" class=\"label\" font-size=\"82\"> Each of the threads subtracts one pixel</text>\n",20,offY-30 );


      fprintf(fp,"</svg>\n");
      fclose(fp);
      return 1;
    }



 return 0;
}*/


int testSetArray(unsigned int * arr, unsigned int width ,unsigned int height,unsigned int value)
{
  unsigned int * arrPTR = arr;
  unsigned int * arrEnd= arr + width * height;
 // unsigned int inc=0;

  fprintf(stderr,"testSetArray (%ux%u) to %u  \n",width,height,value);
  while (arrPTR<arrEnd)
  {
   //fprintf(stderr,"%u ",inc); ++inc;
   *arrPTR = value;
    ++arrPTR;
  }
 return 1;
}


int doCPUonly(int argc, char **argv)
{
    // Load image data
    //allocate mem for the images on host side
    //initialize pointers to NULL to request lib call to allocate as needed
    // PPM images are loaded into 4 byte/pixel memory (RGBX)
    unsigned char *h_needle   = NULL;
    unsigned int needleWidth, needleHeight;

    unsigned char *h_haystack = NULL;
    unsigned int haystackWidth, haystackHeight;

    char *needle   = sdkFindFilePath("needle2.pnm", argv[0]);
    char *haystack = sdkFindFilePath("haystack.pnm", argv[0]);

    printf("Loaded <%s> needle\n", needle);
    if (!sdkLoadPPM4ub(needle, &h_needle, &needleWidth, &needleHeight))    { fprintf(stderr, "Failed to load <%s>\n", needle); }

    printf("Loaded <%s> haystack\n", haystack);
    if (!sdkLoadPPM4ub(haystack, &h_haystack, &haystackWidth, &haystackHeight))    { fprintf(stderr, "Failed to load <%s>\n", haystack); }

    testSetArray((unsigned int*) h_needle,needleWidth,needleHeight,1);
    testSetArray((unsigned int*) h_haystack,haystackWidth,haystackHeight,2);


    unsigned int haystackTilesX = 16;
    unsigned int haystackTilesY = 16;


    //allocate mem for the result on host side
    unsigned int *h_odata = (unsigned int *)malloc( sizeof(unsigned int ) * haystackTilesX * haystackTilesY);
    memset(h_odata,0, sizeof(unsigned int ) * haystackTilesX * haystackTilesY );

    printf("Performing CPU  search...\n");
    printf("Needle Dimensions %ux%u ...\n",needleWidth,needleHeight);
    printf("Haystack Dimensions %ux%u ...\n",haystackWidth,haystackHeight);

    unsigned long startTimer = GetTickCountMilliseconds();
    compareImagesCPU(
                     h_needle,
                     needleWidth,
                     needleHeight,

                     h_haystack,
                     haystackWidth,
                     haystackHeight,

                     haystackTilesX,
                     haystackTilesY,

                     h_odata
                    );
    unsigned long endTimer = GetTickCountMilliseconds();


    //---------------------------------------------------------------------------------------------------
    unsigned int bestID=666 , bestScore=9999999;
    printf("Results are ...\n");
    for (int i=0; i<haystackTilesX*haystackTilesY; i++)
    {
      printf("R[%u]=%u ",i,h_odata[i]);
      if (h_odata[i]<bestScore)
      {
        bestScore = h_odata[i];
        bestID = i;
      }
    }
    printf("\n\n\n");

    printf("Best candidate is %u with score %u ...\n",bestID,bestScore);
    printf("Found it in %lu milliseconds ...\n",endTimer-startTimer);
    //---------------------------------------------------------------------------------------------------

    if (h_odata != NULL)     free(h_odata);
    if (h_needle != NULL)    free(h_needle);
    if (h_haystack != NULL)  free(h_haystack);

 return 0;
}





int queryGPUIsOk(int argc, char **argv)
{
    cudaDeviceProp deviceProp;
    deviceProp.major = 0;
    deviceProp.minor = 0;
    int dev = 0;

    // This will pick the best possible CUDA capable device
    dev = findCudaDevice(argc, (const char **)argv);

    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    // Statistics about the GPU device
    printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
           deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);




    printf("  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)\n",
               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
        printf("  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d layers\n",
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);


        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");









    int version = (deviceProp.major * 0x10 + deviceProp.minor);

    if (version < 0x20)
    {
        printf("Program requires a minimum CUDA compute 2.0 capability\n");
        exit(EXIT_SUCCESS);
        return 0;
    }

return 1;
}



int doGPUonly(int argc, char **argv)
{

    // Load image data
    //allocate mem for the images on host side
    //initialize pointers to NULL to request lib call to allocate as needed
    // PPM images are loaded into 4 byte/pixel memory (RGBX)
    unsigned char *h_needle = NULL;
    unsigned int needleWidth=0, needleHeight=0 , needleSize=0;

    unsigned char *h_haystack = NULL;
    unsigned int haystackWidth=0, haystackHeight=0 ,haystackSize=0;

    char *needle   = sdkFindFilePath("needle2.pnm", argv[0]);
    char *haystack = sdkFindFilePath("haystack.pnm", argv[0]);

    printf("Loaded <%s> needle\n", needle);
    if (!sdkLoadPPM4ub(needle, &h_needle, &needleWidth, &needleHeight))            { fprintf(stderr, "Failed to load <%s>\n", needle);  return 0; }
    needleSize=sizeof(char) *needleWidth*needleHeight * 4;

    printf("Loaded <%s> haystack\n", haystack);
    if (!sdkLoadPPM4ub(haystack, &h_haystack, &haystackWidth, &haystackHeight))    { fprintf(stderr, "Failed to load <%s>\n", haystack); return 0; }
    haystackSize=sizeof(char) * haystackWidth*haystackHeight * 4;

    testSetArray((unsigned int*) h_needle,needleWidth,needleHeight,1);
    testSetArray((unsigned int*) h_haystack,haystackWidth,haystackHeight,2);

    unsigned int haystackTilesX = 16;
    unsigned int haystackTilesY = 16;

    //allocate mem for the result on host side

    //Each of the
    unsigned int blocksPerTileX = (unsigned int) needleWidth/threadSize_x;
    unsigned int blocksPerTileY = (unsigned int) needleHeight/threadSize_y;
    unsigned int resultsPerTile =  blocksPerTileX + blocksPerTileY;
    printf("We expect %ux%u x %u results\n", haystackTilesX, haystackTilesY,resultsPerTile);
    unsigned int odataSize = sizeof(unsigned int) * haystackTilesX*haystackTilesY * resultsPerTile;

    unsigned int *h_odata = (unsigned int *)malloc(odataSize);
    if (h_odata==0) { fprintf(stderr, "Failed to allocate output\n"); return 0; }
    memset(h_odata,0, odataSize);
    //-----------------------------------------------------------------------------

    unsigned int bothDataSize = sizeof(unsigned int) * haystackTilesX*haystackTilesY * resultsPerTile;
    unsigned int *h_bothData = (unsigned int *)malloc(bothDataSize);
    if (h_bothData==0) { fprintf(stderr, "Failed to allocate output\n"); return 0; }
    memset(h_bothData,0, bothDataSize);
    //-----------------------------------------------------------------------------


    unsigned int missDataSize = sizeof(unsigned int) * haystackTilesX*haystackTilesY * resultsPerTile;
    unsigned int *h_missData = (unsigned int *)malloc(missDataSize);
    if (h_missData==0) { fprintf(stderr, "Failed to allocate output\n"); return 0; }
    memset(h_missData,0, missDataSize);
    //-----------------------------------------------------------------------------


    fprintf(stderr,"We will use %ux%u threads\n",threadSize_x,threadSize_y);
    dim3 numThreads = dim3(threadSize_x, threadSize_y, 1);
    dim3 numBlocks  = dim3(iDivUp( haystackWidth, numThreads.x), iDivUp(haystackHeight, numThreads.y));
    fprintf(stderr,"Which means %ux%u blocks\n",numBlocks.x,numBlocks.y);

    /*
    writeCUDALayoutToSVG("gpuCompare.svg",
                         haystackWidth,haystackHeight,
                         needleWidth  ,needleHeight,
                         numBlocks.x,  numBlocks.y,
                         numThreads.x, numThreads.y
                         );*/


    //Check if GPU is ok to proceede
    if (!queryGPUIsOk(argc,argv))
    {
        return 0;
    }

    // allocate device memory for result
    unsigned int *d_odata, *d_bothData, *d_missData , *d_needle, *d_haystack;// , * d_haystackDiffed;

    checkCudaErrors(cudaMalloc((void **) &d_needle, needleSize));
    checkCudaErrors(cudaMalloc((void **) &d_haystack, haystackSize ));
    checkCudaErrors(cudaMalloc((void **) &d_odata, odataSize));
    checkCudaErrors(cudaMalloc((void **) &d_bothData, bothDataSize));
    checkCudaErrors(cudaMalloc((void **) &d_missData, missDataSize));

    // copy host memory to device to initialize to zeros
    checkCudaErrors(cudaMemcpy(d_needle,    h_needle,   needleSize,   cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_haystack,  h_haystack, haystackSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata,     h_odata,    odataSize,    cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_bothData,  h_bothData, bothDataSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_missData,  h_missData, missDataSize, cudaMemcpyHostToDevice));


    printf("Done with copies..\n");
    size_t offset = 0;
    cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<unsigned int>();
    cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<unsigned int>();

    tex2Dneedle.addressMode[0] = cudaAddressModeClamp;
    tex2Dneedle.addressMode[1] = cudaAddressModeClamp;
    tex2Dneedle.filterMode     = cudaFilterModePoint;
    tex2Dneedle.normalized     = false;

    tex2Dhaystack.addressMode[0] = cudaAddressModeClamp;
    tex2Dhaystack.addressMode[1] = cudaAddressModeClamp;
    tex2Dhaystack.filterMode     = cudaFilterModePoint;
    tex2Dhaystack.normalized     = false;


    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dneedle,   d_needle,   ca_desc0,   needleWidth,    needleHeight,   needleWidth*4 ));
    assert(offset == 0);

    printf("Creating texture %ux%u , size %u \n",haystackWidth , haystackHeight, haystackSize);
    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dhaystack, d_haystack, ca_desc1,   haystackWidth , haystackHeight, haystackWidth*4));
    assert(offset == 0);


    printf("Launching CUDA compareImagesKernel warmup()\n");
    // First run the warmup kernel (which we'll use to get the GPU in the correct max power state
    compareImagesKernel<<<numBlocks, numThreads>>>(
                                                    d_needle,
                                                    needleWidth,  needleHeight,

                                                    d_haystack,
                                                    haystackWidth,haystackHeight,


                                                    d_odata,
                                                    d_bothData,
                                                    d_missData,

                                                    1000 //Maximum Difference allowed
                                                   );
    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    printf("Launching CUDA compareImagesKernel()\n");

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // launch the stereoDisparity kernel
    compareImagesKernel<<<numBlocks, numThreads>>>(
                                                    d_needle,
                                                    needleWidth,  needleHeight,

                                                    d_haystack,
                                                    haystackWidth,haystackHeight,


                                                    d_odata ,
                                                    d_bothData,
                                                    d_missData,

                                                    100 //Maximum Difference allowed
                                                   );


    printf("Copying back results\n");
    //Copy result from device to host for verification
    checkCudaErrors(cudaMemcpy(h_odata,    d_odata,    odataSize,    cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_bothData, d_bothData, bothDataSize, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_missData, d_missData, missDataSize, cudaMemcpyDeviceToHost));


    printf("Reducing 4 results into one\n");
    reduceThe4ResultsPerTileIntoOneInPlace(h_odata,haystackWidth/needleWidth,haystackHeight/needleHeight,4);
    reduceThe4ResultsPerTileIntoOneInPlace(h_bothData,haystackWidth/needleWidth,haystackHeight/needleHeight,4);
    reduceThe4ResultsPerTileIntoOneInPlace(h_missData,haystackWidth/needleWidth,haystackHeight/needleHeight,4);

    printf("Done..\n");

    // Record the stop event
    checkCudaErrors(cudaEventRecord(stop, NULL));

    // Wait for the stop event to complete
    checkCudaErrors(cudaEventSynchronize(stop));




    //---------------------------------------------------------------------------------------------------
    unsigned int bestID=666 , bestScore=9999999;
    printf("Results are ...\n");
    for (int i=0; i<haystackTilesX*haystackTilesY; i++)
    {
      printf("R[%u]=%u ",i,h_odata[i]);
      if (h_odata[i]<bestScore)
      {
        bestScore = h_odata[i];
        bestID = i;
      }
    }
    printf("\n\n\n");

    printf("Best candidate is %u with score %u ...\n",bestID,bestScore);
    //---------------------------------------------------------------------------------------------------









    // Check to make sure the kernel didn't fail
    getLastCudaError("Kernel execution failed");

    float msecTotal = 0.0f;
    checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));


    printf("Block Number [%d x %d]\n", numBlocks.x , numBlocks.y);
    printf("Thread Number [%d x %d]\n", numThreads.x , numThreads.y);
    printf("Input Size  [%dx%d]\n", haystackWidth, haystackHeight);
    printf("Kernel size [%dx%d]\n", needleWidth, needleHeight);

    printf("GPU processing time : %.4f (ms)\n", msecTotal);
   // printf("Pixel throughput    : %.3f Mpixels/sec\n", ((float)(w *h*1000.f)/msecTotal)/1000000);



    checkCudaErrors(cudaFree(d_odata));
    checkCudaErrors(cudaFree(d_needle));
    checkCudaErrors(cudaFree(d_haystack));

    if (h_odata != NULL) free(h_odata);
    if (h_needle != NULL)  free(h_needle);
    if (h_haystack != NULL)  free(h_haystack);


 return 0;
}












////////////////////////////////////////////////////////////////////////////////
//! CUDA Sample for calculating depth maps
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    doCPUonly(argc,argv);
    doGPUonly(argc,argv);
    return;


}
