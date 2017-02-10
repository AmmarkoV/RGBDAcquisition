#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "BlobDetector.h"
#include "../../tools/ImageOperations/imageOps.h"



typedef struct { float x, y; } xy;

typedef struct {
                 unsigned int listLength;
                 struct xy* list;
                } xyList;



struct xyList * extractBlobsFromDepthMap(unsigned short * depth , unsigned int width , unsigned int height , unsigned int maxBlobs)
{
  unsigned short * depthPTR = depth;
  unsigned short * depthLimit = depth + (width*height);
  unsigned int lineOffset = (width);
  unsigned short * depthLineLimit = depth + lineOffset;

  unsigned int x=0;
  unsigned int y=0;

  while (depthPTR<depthLimit)
  {
    while (depthPTR<depthLineLimit)
     {
      if (*depthPTR>0)
       {
        fprintf(stderr,"Found blob @ %ux%u \n" ,x,y);
       }
      ++x;
      ++depthPTR;
     }
    depthLineLimit+=lineOffset;
    ++y;
    x=0;

  }


 return 0;
}









int initArgs_BlobDetector(int argc, char *argv[])
{
 return 0;

}

int setConfigStr_BlobDetector(char * label,char * value)
{
 return 0;

}

int setConfigInt_BlobDetector(char * label,int value)
{
 return 0;

}



unsigned char * getDataOutput_BlobDetector(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;
}

int addDataInput_BlobDetector(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{
 fprintf(stderr,"addDataInput_BlobDetector %u (%ux%u)\n" , stream , width, height);
 if (stream==1)
 {
  extractBlobsFromDepthMap( (unsigned short *) data,width,height,10);
 }
 return 0;
}



unsigned short * getDepth_BlobDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}


unsigned char * getColor_BlobDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}



int processData_BlobDetector()
{
 return 0;

}


int cleanup_BlobDetector()
{

 return 0;
}


int stop_BlobDetector()
{
 return 0;

}

