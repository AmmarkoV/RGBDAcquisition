
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imageConversions.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize.h"

#include "fp16.h"

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

  if (fpA!=0)
     {
       fprintf(stderr,"Loaded %u labels \n",labels->numberOfLabels);
       fclose(fpA);
       return 1;
     }
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
	int width=bufW, height=bufH, i;
	const unsigned char *img = (const unsigned char *) buf;
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





float *LoadImageFromMemory32(const char *buf , unsigned int bufW, unsigned int bufH  , unsigned int bufChans, int reqsizeX, int reqsizeY, float *mean)
{
	int width=bufW, height=bufH, i ;
	const unsigned char *img = (const unsigned char *) buf;
	unsigned char *imgresized;
	float *imgfp32;

	if(!img)
	{
		printf("The picture could not be loaded\n");
		return 0;
	}
	imgresized = (unsigned char*) malloc(3*reqsizeX*reqsizeY);
	if(!imgresized)
	{
		perror("malloc");
		return 0;
	}
	stbir_resize_uint8(img, width, height, 0, imgresized, reqsizeX, reqsizeY, 0, 3);

    unsigned int imageSize = sizeof(*imgfp32) * reqsizeX * reqsizeY * 3;
	imgfp32 = (float*) malloc(imageSize);
	if(!imgfp32)
	{
		free(imgresized);
		perror("malloc");
		return 0;
	}
	for(i = 0; i < reqsizeX * reqsizeY * 3; i++)
		imgfp32[i] = imgresized[i];
	free(imgresized);

    if (mean!=0)
    {
	 for(i = 0; i < reqsizeX*reqsizeY; i++)
	  {
	    //flip RGB->BGE and subtract mean
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
    } else
    {
      for(i = 0; i < reqsizeX*reqsizeY; i++)
	  {
      //Just flip RGB->BGE
      float blue, green, red;
	  blue = imgfp32[3*i+2];
	  green = imgfp32[3*i+1];
	  red = imgfp32[3*i+0];

		 imgfp32[3*i+0] = blue;
		 imgfp32[3*i+1] = green;
		 imgfp32[3*i+2] = red;
      }
    }


	return imgfp32;
}
