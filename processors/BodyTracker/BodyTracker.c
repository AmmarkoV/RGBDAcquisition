#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "BodyTracker.h"
#include "../../tools/ImageOperations/imageOps.h"


#include "forth_skeleton_tracker_redist/headers/FORTHUpperBodyGestureTrackerLib.h"

unsigned int framesProcessed=0;
unsigned char * colorFrame;
unsigned int colorWidth;
unsigned int colorHeight;

unsigned short * depthFrame;
unsigned int depthWidth;
unsigned int depthHeight;

struct calibrationFUBGT frameCalibration={0};

int initArgs_BodyDetector(int argc, char *argv[])
{
 fprintf(stdout,"initArgs_BodyDetector\n");

 fubgtUpperBodyTracker_Initialize(640,480,"../",argc,argv);
 //fubgtUpperBodyTracker_setVisualization(1);
 return 0;

}

int setConfigStr_BodyDetector(char * label,char * value)
{
 return 0;

}

int setConfigInt_BodyDetector(char * label,int value)
{
 return 0;

}



unsigned char * getDataOutput_BodyDetector(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;
}

int addDataInput_BodyDetector(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{

 fprintf(stderr,"addDataInput_BodyDetector %u (%ux%u)\n" , stream , width, height);

 if (stream==0)
 {
    depthFrame = (unsigned short*) data;
    depthWidth = width;
    depthHeight = height;
 } else
 if (stream==1)
 {
    colorFrame = (char*) data;
    colorWidth = width;
    colorHeight = height;

   ++framesProcessed;

  return 1;
 }


 return 0;
}



unsigned short * getDepth_BodyDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}

unsigned char * getColor_BodyDetector(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;

}

int processData_BodyDetector()
{

   fubgtUpperBodyTracker_NewFrame( colorFrame, colorWidth, colorHeight,
                                   depthFrame , depthWidth , depthHeight,
                                   &frameCalibration,
                                   0,
                                   framesProcessed);

 return 0;

}


int cleanup_BodyDetector()
{

 return 0;
}


int stop_BodyDetector()
{
 fubgtUpperBodyTracker_Close();
 return 0;
}

