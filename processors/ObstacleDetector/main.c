#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "ObstacleDetector.h"
#include "../../tools/ImagePrimitives/image.h"
#include "../../tools/Codecs/codecs.h"



struct Image * mask={0};


int initArgs_ObstacleDetector(int argc, char *argv[])
{
  mask=readImage("processors/ObstacleDetector/corridorMask.png",PNG_CODEC,0);
}

int setConfigStr_ObstacleDetector(char * label,char * value)
{

}


int setConfigInt_ObstacleDetector(char * label,int value)
{

}


unsigned char * getDataOutput_ObstacleDetector(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{

}


int addDataInput_ObstacleDetector(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{

}


unsigned short * getDepth_ObstacleDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{

}


unsigned char * getColor_ObstacleDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{

}


int processData_ObstacleDetector()
{

}

int cleanup_ObstacleDetector()
{

}

int stop_ObstacleDetector()
{
 destroyImage(mask);
}
