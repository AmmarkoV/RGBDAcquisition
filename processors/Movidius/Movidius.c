// Copyright 2017 Intel Corporation.
// The source code, information and material ("Material") contained herein is
// owned by Intel Corporation or its suppliers or licensors, and title to such
// Material remains with Intel Corporation or its suppliers or licensors.
// The Material contains proprietary information of Intel or its suppliers and
// licensors. The Material is protected by worldwide copyright laws and treaty
// provisions.
// No part of the Material may be used, copied, reproduced, modified, published,
// uploaded, posted, transmitted, distributed or disclosed in any way without
// Intel's prior express written permission. No license under any patent,
// copyright or other intellectual property rights in the Material is granted to
// or conferred upon you, either expressly, by implication, inducement, estoppel
// or otherwise.
// Any license under such intellectual property rights must be express and
// approved by Intel in writing.

#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include "fp16.h"
#include <mvnc.h>


// somewhat arbitrary buffer size for the device name
#define NAME_SIZE 100

// graph file name - assume we are running in this directory: ncsdk/examples/caffe/GoogLeNet/cpp
#define WORDS_FILE_NAME "../processors/Movidius/googlenet/words.txt"
#define GRAPH_FILE_NAME "../processors/Movidius/googlenet/graph"



// 16 bits.  will use this to store half precision floats since C++ has no
// built in support for it.
typedef unsigned short half;

// GoogleNet image dimensions, network mean values for each channel in BGR order.
const int networkDim = 224;
float networkMean[] = {0.40787054*255.0, 0.45752458*255.0, 0.48109378*255.0};


struct labelContents
{
  unsigned int numberOfLabels;
  char ** content;
};

struct labelContents labels={0};



struct movidiusContext
{
  const int networkDim;
  float networkMean[3];

  ncStatus_t retCode;
  struct ncDeviceHandle_t * deviceHandle;
  char devName[NAME_SIZE];

  unsigned int graphFileLen;
  void* graphFileBuf;


  struct ncGraphHandle_t *graphHandle;

  struct ncFifoHandle_t * bufferIn;
  struct ncFifoHandle_t * bufferOut;

  struct ncTensorDescriptor_t inputTensorDesc;
  struct ncTensorDescriptor_t outputTensorDesc;

  //void* graphHandle;
};

struct movidiusContext mov={0};




int loadLabels(const char * filename , struct labelContents * labels )
{
  labels->content = (char ** ) malloc(sizeof(char ** ) * 10000);

  ssize_t readA;

  FILE * fpA = fopen(filename,"r");
  if  (fpA!=0)
  {

    char * lineA = NULL;
    size_t lenA = 0;

    while  ((readA = getline(&lineA, &lenA, fpA)) != -1)
    {
      lineA[strcspn(lineA, "\r\n")] = 0; // works for LF, CR, CRLF, LFCR, ...


      labels->content[labels->numberOfLabels]=lineA;
      ++labels->numberOfLabels;
      lineA=0;
    }

    //if (lineA) { free(lineA); }
  }

  if (fpA!=0) { fclose(fpA); return 1; }
 return 0;
}



// Load a graph file
// caller must free the buffer returned.
void *LoadFile(const char *path, unsigned int *length)
{
	FILE *fp;
	char *buf;

	fp = fopen(path, "rb");
	if(fp == NULL)
		return 0;
	fseek(fp, 0, SEEK_END);
	*length = ftell(fp);
	rewind(fp);
	if(!(buf = (char*) malloc(*length)))
	{
		fclose(fp);
		return 0;
	}
	if(fread(buf, 1, *length, fp) != *length)
	{
		fclose(fp);
		free(buf);
		return 0;
	}
	fclose(fp);
	return buf;
}


half *LoadImage(const char *path, int reqsize, float *mean)
{
	int width, height, cp, i;
	unsigned char *img, *imgresized;
	float *imgfp32;
	half *imgfp16;

	img = stbi_load(path, &width, &height, &cp, 3);
	if(!img)
	{
		printf("The picture %s could not be loaded\n", path);
		return 0;
	}
	imgresized = (unsigned char*) malloc(3*reqsize*reqsize);
	if(!imgresized)
	{
		free(img);
		perror("malloc");
		return 0;
	}
	stbir_resize_uint8(img, width, height, 0, imgresized, reqsize, reqsize, 0, 3);
	free(img);
	imgfp32 = (float*) malloc(sizeof(*imgfp32) * reqsize * reqsize * 3);
	if(!imgfp32)
	{
		free(imgresized);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsize * reqsize * 3; i++)
		imgfp32[i] = imgresized[i];
	free(imgresized);
	imgfp16 = (half*) malloc(sizeof(*imgfp16) * reqsize * reqsize * 3);
	if(!imgfp16)
	{
		free(imgfp32);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsize*reqsize; i++)
	{
		float blue, green, red;
                blue = imgfp32[3*i+2];
                green = imgfp32[3*i+1];
                red = imgfp32[3*i+0];

                imgfp32[3*i+0] = blue-mean[0];
                imgfp32[3*i+1] = green-mean[1];
                imgfp32[3*i+2] = red-mean[2];

                // uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
                //printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
	}
	floattofp16((unsigned char *)imgfp16, imgfp32, 3*reqsize*reqsize);
	free(imgfp32);
	return imgfp16;
}





half *LoadImageFromMemory16(const char *buf , unsigned int bufW, unsigned int bufH  , unsigned int bufChans, int reqsize, float *mean)
{
	int width=bufW, height=bufH, cp=bufChans, i;
	const char *img = buf;
	unsigned char *imgresized;
	float *imgfp32;
	half *imgfp16;

	if(!img)
	{
		printf("The picture could not be loaded\n");
		return 0;
	}
	imgresized = (unsigned char*) malloc(3*reqsize*reqsize);
	if(!imgresized)
	{
		//free(img);
		perror("malloc");
		return 0;
	}
	stbir_resize_uint8(img, width, height, 0, imgresized, reqsize, reqsize, 0, 3);
	//free(img);
	imgfp32 = (float*) malloc(sizeof(*imgfp32) * reqsize * reqsize * 3);
	if(!imgfp32)
	{
		free(imgresized);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsize * reqsize * 3; i++)
		imgfp32[i] = imgresized[i];
	free(imgresized);
	imgfp16 = (half*) malloc(sizeof(*imgfp16) * reqsize * reqsize * 3);
	if(!imgfp16)
	{
		free(imgfp32);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsize*reqsize; i++)
	{
		float blue, green, red;
                blue = imgfp32[3*i+2];
                green = imgfp32[3*i+1];
                red = imgfp32[3*i+0];

                imgfp32[3*i+0] = blue-mean[0];
                imgfp32[3*i+1] = green-mean[1];
                imgfp32[3*i+2] = red-mean[2];

                // uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
                //printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
	}
	floattofp16((unsigned char *)imgfp16, imgfp32, 3*reqsize*reqsize);
	free(imgfp32);
	return imgfp16;
}





float *LoadImageFromMemory32(const char *buf , unsigned int bufW, unsigned int bufH  , unsigned int bufChans, int reqsize, float *mean)
{
	int width=bufW, height=bufH, cp=bufChans, i ;
	unsigned char *img, *imgresized;
	float *imgfp32;
	img = buf;

	if(!img)
	{
		printf("The picture could not be loaded\n");
		return 0;
	}
	imgresized = (unsigned char*) malloc(3*reqsize*reqsize);
	if(!imgresized)
	{
		perror("malloc");
		return 0;
	}
	stbir_resize_uint8(img, width, height, 0, imgresized, reqsize, reqsize, 0, 3);

    unsigned int imageSize = sizeof(*imgfp32) * reqsize * reqsize * 3;
	imgfp32 = (float*) malloc(imageSize);
	if(!imgfp32)
	{
		free(imgresized);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsize * reqsize * 3; i++)
		imgfp32[i] = imgresized[i];
	free(imgresized);
	for(i = 0; i < reqsize*reqsize; i++)
	{
		float blue, green, red;
		blue = imgfp32[3*i+2];
		green = imgfp32[3*i+1];
		red = imgfp32[3*i+0];

		imgfp32[3*i+0] = blue-mean[0];
		imgfp32[3*i+1] = green-mean[1];
		imgfp32[3*i+2] = red-mean[2];

		// uncomment to see what values are getting passed to mvncLoadTensor() before conversion to half float
		//printf("Blue: %f, Grean: %f,  Red: %f \n", imgfp32[3*i+0], imgfp32[3*i+1], imgfp32[3*i+2]);
	}
	return imgfp32;
}










int initArgs_Movidius(int argc, char *argv[])
{
    if (!loadLabels(WORDS_FILE_NAME,&labels )) { exit(0); }

    int loglevel = 2;
    mov.retCode = ncGlobalSetOption(NC_RW_LOG_LEVEL, &loglevel, sizeof(loglevel));

    mov.retCode = ncDeviceCreate(0, &mov.deviceHandle);
    if(mov.retCode != NC_OK)
    {
        printf("Error - No NCS devices found.\n");
        printf("    ncStatus value: %d\n", mov.retCode);
        return 0;
    }

    // Try to open the NCS device via the device name
    mov.retCode = ncDeviceOpen(mov.deviceHandle);
    if (mov.retCode != NC_OK)
    {   // failed to open the device.
        printf("Could not open NCS device\n");
        return 0;
    }

    // deviceHandle is ready to use now.
    // Pass it to other NC API calls as needed and close it when finished.
    printf("Successfully opened NCS device!\n");


    // Now read in a graph file
    mov.graphFileBuf = LoadFile(GRAPH_FILE_NAME, &mov.graphFileLen);

    // Init graph handle
    mov.retCode = ncGraphCreate("graph", &mov.graphHandle);
    if (mov.retCode != NC_OK)
    {
        printf("Error - ncGraphCreate failed\n");
        return 0;
    }

    // Send graph to device
    mov.retCode = ncGraphAllocate(mov.deviceHandle, mov.graphHandle, mov.graphFileBuf, mov.graphFileLen);
    if (mov.retCode != NC_OK)
    {   // error allocating graph
        printf("Could not allocate graph for file: %s\n", GRAPH_FILE_NAME);
        printf("Error from ncGraphAllocate is: %d\n", mov.retCode);
    }
    else
    {
        // successfully allocated graph.  Now graphHandle is ready to go.
        // use graphHandle for other API calls and call mvncDeallocateGraph
        // when done with it.
        printf("Successfully allocated graph for %s\n", GRAPH_FILE_NAME);

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



int addDataInput_Movidius(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{
 if (stream==0)
 {
        // LoadImage will read image from disk, convert channels to floats
        // subtract network mean for each value in each channel.  Then, convert
        // floats to half precision floats and return pointer to the buffer
        // of half precision floats (Fp16s)
        //half* imageBufFp16 = LoadImage(IMAGE_FILE_NAME, networkDim, networkMean);
        //half* imageBufFp16 = LoadImageFromMemory16( (const char* ) data , width ,height , channels, networkDim, networkMean);

        float * imageBufFp32 = LoadImageFromMemory32( (const char* ) data , width ,height , channels, networkDim, networkMean);


        if (imageBufFp32!=0)
        {
        // calculate the length of the buffer that contains the half precision floats.
        // 3 channels * width * height * sizeof a 16bit float
        unsigned int lenBufFp32 = 3*networkDim*networkDim*sizeof(*imageBufFp32);


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
                //printf("resultData length is %d \n", numResults);
                float *fresult = (float*) result;

                float maxResult = 0.0;
                int maxIndex = -1;
                for (int index = 0; index < numResults; index++)
                {
                    // printf("Category %d is: %f\n", index, resultData32[index]);
                    if (fresult[index] > maxResult)
                    {
                        maxResult = fresult[index];
                        maxIndex = index;
                    }
                }


                printf("Index of top result is: %d\n", maxIndex);
                printf("Probability of top result is: %f\n", fresult[maxIndex]);
                if (labels.content[maxIndex]!=0 ) { printf("This is %s \n",labels.content[maxIndex]); }
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


  free(mov.graphFileBuf);
  mov.retCode = ncDeviceClose(mov.deviceHandle);
  mov.deviceHandle = NULL;

 return 1;
}










