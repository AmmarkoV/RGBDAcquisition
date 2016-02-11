#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ObstacleDetector.h"
#include "../ViewpointChange/viewpoint_change.h"

#include "../../tools/Primitives/image.h"
#include "../../tools/Codecs/codecs.h"

unsigned int bleeps=0;

unsigned char * colorFrame = 0;
unsigned int colorWidth=0,colorHeight=0,colorChannels=0,colorBitsperpixel=0,colorFrameSize=0;

unsigned short * depthFrame = 0;
unsigned int depthWidth=0,depthHeight=0,depthChannels=0,depthBitsperpixel=0,depthFrameSize=0;

struct Image * mask={0};

int doBeep()
{
 int i=system("paplay ../processors/ObstacleDetector/bleep.wav&");
 return (i==0);
}


int initArgs_ObstacleDetector(int argc, char *argv[])
{
  mask=readImage("../processors/ObstacleDetector/corridorMask.png",PNG_CODEC,0);
  return 1;
}

int setConfigStr_ObstacleDetector(char * label,char * value)
{
 return 0;
}


int setConfigInt_ObstacleDetector(char * label,int value)
{
 return 0;
}


unsigned char * getDataOutput_ObstacleDetector(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;
}


int addDataInput_ObstacleDetector(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{
  if (stream==0)
  {
    unsigned int bytesperpixel = (unsigned int) bitsperpixel/8;
    colorFrameSize = width*height*channels*bytesperpixel;
    //fprintf(stderr,"Color Frame %ux%u channels %u , bitsperpixel %u ( size = %u ) \n",width,height,channels ,bitsperpixel , colorFrameSize);
    colorFrame = (unsigned char* ) malloc(colorFrameSize);
    if (colorFrame!=0)
    {
      memcpy(colorFrame,data,colorFrameSize);
      colorWidth=width; colorHeight=height;  colorChannels=channels; colorBitsperpixel=bitsperpixel;
    }
    return 1;
  } else
  if (stream==1)
  {
    unsigned int bytesperpixel = (unsigned int) bitsperpixel/8;
    depthFrameSize = width*height*channels*bytesperpixel;
    //fprintf(stderr,"Depth Frame %ux%u channels %u , bitsperpixel %u ( size = %u ) \n",width,height,channels ,bitsperpixel , depthFrameSize);
    depthFrame = (unsigned short* ) malloc(depthFrameSize);
    if (depthFrame!=0)
    {
      memcpy(depthFrame,data,depthFrameSize);
      depthWidth=width; depthHeight=height;  depthChannels=channels; depthBitsperpixel=bitsperpixel;
    }
    return 1;
   }


 return 0;

}


unsigned short * getDepth_ObstacleDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 *width=depthWidth; *height=depthHeight; *channels=depthChannels;  *bitsperpixel=depthBitsperpixel;
 return depthFrame;
}


unsigned char * getColor_ObstacleDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 *width=colorWidth; *height=colorHeight; *channels=colorChannels;  *bitsperpixel=colorBitsperpixel;
 return colorFrame;
}


int processData_ObstacleDetector()
{
  unsigned int DEPTH_RANGE = 10000;
  unsigned char * bev  =0;
  bev = birdsEyeView(colorFrame, depthFrame, colorWidth,colorHeight, 0,DEPTH_RANGE);
  memcpy(colorFrame,bev , colorWidth *colorHeight*3);


  if (bev!=0)
         {
           //if(VIEW_SITUATION) { viewImage("bevFrame",&bevImg);  }
           unsigned int fitScore = FitImageInMask(bev,mask->pixels,mask->width,mask->height);
           fprintf(stderr,"Got a fit of %u\n",fitScore);
           if (fitScore < 9000)
           {
             ++bleeps;
             if (bleeps%10==0) {  doBeep(); }
           }

           free(bev);
           return 1;
        } else
        { fprintf(stderr,"Could not perform a birdseyeview translation\n"); }


  return 0;
}

int cleanup_ObstacleDetector()
{
    if (colorFrame!=0) { free(colorFrame); colorFrame=0; }
    if (depthFrame!=0) { free(depthFrame); depthFrame=0; }
 return 1;
}

int stop_ObstacleDetector()
{
 destroyImage(mask);
 return 1;
}
