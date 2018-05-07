//This code is loosely  based on intel's examples
//For the movidius NCS SDK
//https://github.com/movidius/ncsdk/tree/master/examples

#include <stdio.h>
#include <stdlib.h>
#include <mvnc.h>
#include "imageConversions.h"

#include "MovidiusTypes.h"
#include "tinyYolo.h"



#define USE_GOOGLENET 0





#if USE_GOOGLENET
// GoogleNet image dimensions, network mean values for each channel in BGR order.
float minimumConfidence = 0.4;
const int networkDimX = 224;
const int networkDimY = 224;
int packedInfo = 0;
float networkMean[] = {0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0};
#define WORDS_FILE_NAME "../processors/Movidius/googlenet/words.txt"
#define GRAPH_FILE_NAME "../processors/Movidius/googlenet/graph"
#else
float minimumConfidence = 0.9;
const int networkDimX = 448;
const int networkDimY = 448;
int packedInfo = 1;
float * networkMean = 0;
#define WORDS_FILE_NAME "../processors/Movidius/tinyyolo/words.txt"
#define GRAPH_FILE_NAME "../processors/Movidius/tinyyolo/graph"
#endif

struct labelContents labels={0};



struct movidiusContext
{
  ncStatus_t retCode;
  struct ncDeviceHandle_t * deviceHandle;

  struct ncDeviceHandle_t * deviceHandle0;
  struct ncDeviceHandle_t * deviceHandle1;
  struct ncDeviceHandle_t * deviceHandle2;

  unsigned int graphFileLen;
  void* graphFileBuf;


  struct ncGraphHandle_t *graphHandle;

  struct ncFifoHandle_t * bufferIn;
  struct ncFifoHandle_t * bufferOut;

  struct ncTensorDescriptor_t inputTensorDesc;
  struct ncTensorDescriptor_t outputTensorDesc;

};

struct movidiusContext mov={0};

// Opens one NCS device.
// Param deviceIndex is the zero-based index of the device to open
// Param deviceHandle is the address of a device handle that will be set
//                    if opening is successful
// Returns true if works or false if doesn't.
int openOneNCS(int deviceIndex, struct ncDeviceHandle_t **deviceHandle)
{
    ncStatus_t retCode;
    retCode = ncDeviceCreate(deviceIndex, deviceHandle);
    if (retCode != NC_OK)
    {   // failed to get this device's name, maybe none plugged in.
        printf("Error - NCS device at index %d not found\n", deviceIndex);
        return 0;
    }

    // Try to open the NCS device via the device name
    retCode = ncDeviceOpen(*deviceHandle);
    if (retCode != NC_OK)
    {   // failed to open the device.
        printf("Error - Could not open NCS device at index %d\n", deviceIndex);
        return 0;
    }

    // deviceHandle is ready to use now.
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Successfully opened NCS device at index %d %p!\n", deviceIndex,*deviceHandle);
    return 1;
}


// Loads a compiled network graph onto the NCS device.
// Param deviceHandle is the open device handle for the device that will allocate the graph
// Param graphFilename is the name of the compiled network graph file to load on the NCS
// Param graphHandle is the address of the graph handle that will be created internally.
//                   the caller must call mvncDeallocateGraph when done with the handle.
// Returns true if works or false if doesn't.
void * loadGraphToNCS(
                      struct ncDeviceHandle_t* deviceHandle,
                      const char* graphFilename,
                      unsigned int * graphFileLength,
                      struct ncGraphHandle_t** graphHandle
                     )
{
    ncStatus_t retCode;
    int rc = 0;
    unsigned int mem, memMax, length;

    // Read in a graph file
    void* graphFileBuf = LoadFile(graphFilename, graphFileLength);

    length = sizeof(unsigned int);
    // Read device memory info
    rc = ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_CURRENT_MEMORY_USED, (void **)&mem, &length);
    rc += ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_MEMORY_SIZE, (void **)&memMax, &length);
    if(rc)
        printf("ncDeviceGetOption failed, rc=%d\n", rc);
    else
        printf("Current memory used on device is %d out of %d\n", mem, memMax);

    // allocate the graph
    retCode = ncGraphCreate("graph", graphHandle);
    if (retCode)
    {
        printf("ncGraphCreate failed, retCode=%d\n", retCode);
        return 0;
    }

    // Send graph to device
    retCode = ncGraphAllocate(deviceHandle, *graphHandle, graphFileBuf, *graphFileLength);
    if (retCode != NC_OK)
    {   // error allocating graph
        printf("Could not allocate graph for file: %s\n", graphFilename);
        printf("Error from ncGraphAllocate is: %d\n", retCode);
        return 0;
    }

    // successfully allocated graph.  Now graphHandle is ready to go.
    // use graphHandle for other API calls and call ncGraphDestroy
    // when done with it.
    printf("Successfully allocated graph for %s\n", graphFilename);

    return graphFileBuf;
}


int initArgs_Movidius(int argc, char *argv[])
{
    if (!loadLabels(WORDS_FILE_NAME,&labels )) { exit(0); }

    int loglevel = 2;
    mov.retCode = ncGlobalSetOption(NC_RW_LOG_LEVEL, &loglevel, sizeof(loglevel));


    if (!openOneNCS(0 /* Device ID for first device */,&mov.deviceHandle))
    {
        printf("Could not open NCS device 0\n");
        return 0;

    }

    mov.graphFileBuf=loadGraphToNCS(mov.deviceHandle,GRAPH_FILE_NAME,&mov.graphFileLen,&mov.graphHandle);
    if (!mov.graphFileBuf)
    {
        printf("Could not allocate graph for file: %s\n", GRAPH_FILE_NAME);
        return 0;
    }
    else
    {
        // successfully allocated graph.  Now graphHandle is ready to go.
        // use graphHandle for other API calls and call mvncDeallocateGraph
        // when done with it.

        // Read tensor descriptors
        unsigned int length = sizeof(struct ncTensorDescriptor_t);
        ncGraphGetOption(mov.graphHandle, NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS, &mov.inputTensorDesc,  &length);
        ncGraphGetOption(mov.graphHandle, NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS, &mov.outputTensorDesc,  &length);
        int dataTypeSize = mov.outputTensorDesc.totalSize/(mov.outputTensorDesc.w* mov.outputTensorDesc.h*mov.outputTensorDesc.c*mov.outputTensorDesc.n);
        printf("output data type size %d \n", dataTypeSize);

        // Init & Create Fifos
        mov.retCode = ncFifoCreate("FifoIn0",NC_FIFO_HOST_WO, &mov.bufferIn);
        if (mov.retCode != NC_OK)
        {
            printf("Error - Input Fifo Initialization failed!");
            return 0;
        }
        mov.retCode = ncFifoAllocate(mov.bufferIn, mov.deviceHandle, &mov.inputTensorDesc, 2);
        if (mov.retCode != NC_OK)
        {
            printf("Error - Input Fifo allocation failed!");
            return 0;
        }
        mov.retCode = ncFifoCreate("FifoOut0",NC_FIFO_HOST_RO, &mov.bufferOut);
        if (mov.retCode != NC_OK)
        {
            printf("Error - Output Fifo Initialization failed!");
            return 0;
        }
        mov.retCode = ncFifoAllocate(mov.bufferOut, mov.deviceHandle, &mov.outputTensorDesc, 2);
        if (mov.retCode != NC_OK)
        {
            printf("Error - Output Fifo allocation failed!");
            return 0;
        }

        // successfully allocated graph.  Now graphHandle is ready to go.
        // use graphHandle for other API calls and call mvncDeallocateGraph
        // when done with it.
        printf("Successfully allocated graph for %s\n", GRAPH_FILE_NAME);
      return 1;
    }
 return 0;
}




int processGoogleNet(float * results , unsigned int resultsLength )
{
  float maxResult = 0.0;
  int maxIndex = -1;
  for (int index = 0; index < resultsLength; index++)
                {
                    // printf("Category %d is: %f\n", index, resultData32[index]);
                    if (results[index] > maxResult)
                    {
                        maxResult = results[index];
                        maxIndex = index;
                    }
                }

   if (results[maxIndex]>=minimumConfidence)
                {
                  printf("Index of top result is: %d\n", maxIndex);
                  printf("Probability of top result is: %f\n", results[maxIndex]);
                  if (maxIndex<labels.numberOfLabels)
                   {
                    if (labels.content[maxIndex]!=0 )
                        { printf("This is %s \n",labels.content[maxIndex]); }
                   } else
                   { printf("Incorrect result(?) \n"); }
                }
 return 1;
}



int addDataInput_Movidius(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{
 if (stream==0)
 {
        // LoadImage will read image from disk, convert channels to floats
        // subtract network mean for each value in each channel.  Then, convert
        // floats to half precision floats and return pointer to the buffer
        // of half precision floats (Fp16s)
        //half* imageBufFp16 = LoadImage(IMAGE_FILE_NAME, networkDim, networkMean);
        //half* imageBufFp16 = LoadImageFromMemory16( (const char* ) data , width ,height , channels, networkDimX , networkDimY, networkMean);

        float * imageBufFp32 = LoadImageFromMemory32( (const char* ) data , width ,height , channels, networkDimX , networkDimY , networkMean);


        if (imageBufFp32!=0)
        {
        // calculate the length of the buffer that contains the half precision floats.
        // 3 channels * width * height * sizeof a 16bit float
        unsigned int lenBufFp32 = 3*networkDimX*networkDimY*sizeof(*imageBufFp32);


        // Write tensor to input fifo
        mov.retCode = ncFifoWriteElem(mov.bufferIn, imageBufFp32, &lenBufFp32, 0);
        if (mov.retCode != NC_OK)
        {
            printf("Error - Failed to write element to Fifo!\n");
            free(imageBufFp32);
            return 0;
        }
        else
        {   // the inference has been started, now call mvncGetResult() for the
            // inference result
            //printf("Successfully loaded the tensor for image \n");

            // queue inference
            mov.retCode = ncGraphQueueInference(mov.graphHandle, &mov.bufferIn, 1, &mov.bufferOut, 1);
            if (mov.retCode != NC_OK)
            {
                printf("Error - Failed to queue Inference!");
                free(imageBufFp32);
                return 0;
            }
            //free(image);
            // Read output results
            unsigned int outputDataLength=0;
            unsigned int length = sizeof(unsigned int);
            mov.retCode = ncFifoGetOption(mov.bufferOut, NC_RO_FIFO_ELEMENT_DATA_SIZE, &outputDataLength, &length);
            if (mov.retCode || length != sizeof(unsigned int))
            {
                printf("ncFifoGetOption failed, rc=%d\n", mov.retCode);
                free(imageBufFp32);
                return 0;
            }
            void *result = malloc(outputDataLength);
            if (!result)
            {
                printf("malloc failed!\n");
                free(imageBufFp32);
                return 0;
            }
            void *userParam;
            mov.retCode = ncFifoReadElem(mov.bufferOut, result, &outputDataLength, &userParam);
            if (mov.retCode != NC_OK)
            {
                printf("Error - Read Inference result failed!");
                return 0;
            }
            else
            {   // Successfully got the result.  The inference result is in the buffer pointed to by resultData
                //printf("Successfully got the inference result for image \n");
                unsigned int numResults =  outputDataLength / sizeof(float);
                float *fresult = (float*) result;
                //printf("resultData length is %d \n", numResults);
                if (packedInfo)
                {
                  processTinyYOLO(&labels,fresult, numResults , data , width , height, minimumConfidence);
                }  else
                {
                  processGoogleNet(fresult,numResults);
                }



            }
            free(result);
            free(imageBufFp32);
        }


         //free(imageBufFp16);
	     return 1;
        }
  }
  return 0;
}


int stop_Movidius()
{
 mov.retCode = ncGraphDestroy(&mov.graphHandle);
 if (mov.retCode != NC_OK)
        {
            printf("Error - Failed to deallocate graph!");
            exit(-1);
        }

  mov.retCode = ncFifoDestroy(&mov.bufferOut);
  if (mov.retCode != NC_OK)
        {
            printf("Error - Failed to deallocate fifo!");
            exit(-1);
        }

  mov.retCode = ncFifoDestroy(&mov.bufferIn);
  if (mov.retCode != NC_OK)
        {
            printf("Error - Failed to deallocate fifo!");
            exit(-1);
        }
  mov.graphHandle = NULL;


  if (mov.graphFileBuf!=0) { free(mov.graphFileBuf); }
  mov.retCode = ncDeviceClose(mov.deviceHandle);
  mov.deviceHandle = NULL;

 return 1;
}





int setConfigStr_Movidius(char * label,char * value)
{
 return 0;

}

int setConfigInt_Movidius(char * label,int value)
{
 return 0;

}

unsigned char * getDataOutput_Movidius(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;
}

unsigned short * getDepth_Movidius(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}

unsigned char * getColor_Movidius(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}

int processData_Movidius()
{
 return 0;

}

int cleanup_Movidius()
{

 return 0;
}








