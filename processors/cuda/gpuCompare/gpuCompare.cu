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
    unsigned int needleWidth, needleHeight , needleSize;

    unsigned char *h_haystack = NULL;
    unsigned int haystackWidth, haystackHeight ,haystackSize;

    char *needle   = sdkFindFilePath("needle2.pnm", argv[0]);
    char *haystack = sdkFindFilePath("haystack.pnm", argv[0]);

    printf("Loaded <%s> needle\n", needle);
    if (!sdkLoadPPM4ub(needle, &h_needle, &needleWidth, &needleHeight))    { fprintf(stderr, "Failed to load <%s>\n", needle); }
    needleSize=needleWidth*needleHeight*4;

    printf("Loaded <%s> haystack\n", haystack);
    if (!sdkLoadPPM4ub(haystack, &h_haystack, &haystackWidth, &haystackHeight))    { fprintf(stderr, "Failed to load <%s>\n", haystack); }
    haystackSize=haystackWidth*haystackHeight*4;



    unsigned int haystackTilesX = 16;
    unsigned int haystackTilesY = 16;
    //allocate mem for the result on host side
    unsigned int *h_odata = (unsigned int *)malloc( sizeof(unsigned int ) * haystackTilesX * haystackTilesY);
    unsigned int odataSize = haystackTilesX*haystackTilesY;
    memset(h_odata,0, sizeof(unsigned int ) * haystackTilesX * haystackTilesY );

    //initialize the memory
    if (!queryGPUIsOk(argc,argv))
    {
        return 0;
    }


    dim3 numThreads = dim3(blockSize_x, blockSize_y, 1);
    dim3 numBlocks = dim3(iDivUp( haystackWidth, numThreads.x), iDivUp(haystackHeight, numThreads.y));



    // allocate device memory for result
    unsigned int *d_odata, *d_needle, *d_haystack;

    checkCudaErrors(cudaMalloc((void **) &d_odata, odataSize));
    checkCudaErrors(cudaMalloc((void **) &d_needle, needleSize));
    checkCudaErrors(cudaMalloc((void **) &d_haystack, haystackSize ));

    // copy host memory to device to initialize to zeros
    checkCudaErrors(cudaMemcpy(d_needle,    h_needle, needleSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_haystack,  h_haystack, haystackSize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_odata, odataSize, cudaMemcpyHostToDevice));


    printf("Done with copies..\n");
    size_t offset = 0;
    cudaChannelFormatDesc ca_desc0 = cudaCreateChannelDesc<unsigned int>();
    cudaChannelFormatDesc ca_desc1 = cudaCreateChannelDesc<unsigned int>();

    tex2Dleft.addressMode[0] = cudaAddressModeClamp;
    tex2Dleft.addressMode[1] = cudaAddressModeClamp;
    tex2Dleft.filterMode     = cudaFilterModePoint;
    tex2Dleft.normalized     = false;
    tex2Dright.addressMode[0] = cudaAddressModeClamp;
    tex2Dright.addressMode[1] = cudaAddressModeClamp;
    tex2Dright.filterMode     = cudaFilterModePoint;
    tex2Dright.normalized     = false;


    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dleft,  d_needle, ca_desc0,   needleWidth,needleHeight, needleSize ));
    assert(offset == 0);

    checkCudaErrors(cudaBindTexture2D(&offset, tex2Dright, d_haystack, ca_desc1, haystackWidth , haystackHeight, haystackSize));
    assert(offset == 0);

    // First run the warmup kernel (which we'll use to get the GPU in the correct max power state
    printf("Start test run ? \n");
    //compareImagesKernel<<<numBlocks, numThreads>>>(d_needle, d_haystack, d_odata, w, h );
    cudaDeviceSynchronize();

    // Allocate CUDA events that we'll use for timing
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    printf("Launching CUDA compareImagesKernel()\n");

    // Record the start event
    checkCudaErrors(cudaEventRecord(start, NULL));

    // launch the stereoDisparity kernel
    //compareImagesKernel<<<numBlocks, numThreads>>>(d_needle, d_haystack, d_odata, w, h );
    //Copy result from device to host for verification
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, odataSize, cudaMemcpyDeviceToHost));


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


    printf("Input Size  [%dx%d], ", haystackWidth, haystackHeight);
    printf("Kernel size [%dx%d], ", needleWidth, needleHeight);

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
