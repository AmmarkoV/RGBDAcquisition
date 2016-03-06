#include <iostream>
#include "DisparityMapping.h"

#include "sgbm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cv.h"
#include "highgui.h"
#include "cvaux.h"
#include "imageStream.h"
#include "opticalFlow.h"
#include "stereo_calibrate.h"


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

#define UNIFY_LKFLOWS 1


using namespace std;

unsigned char * colorFrame = 0;
unsigned int colorWidth=0,colorHeight=0,colorChannels=0,colorBitsperpixel=0;

unsigned short * depthFrame = 0;
unsigned int depthWidth=0,depthHeight=0,depthChannels=0,depthBitsperpixel=0;

#define DISPARITYMAPPING_STRING_SIZE 512
unsigned int disparityCalibrationUsed=0;
char disparityCalibrationPath[DISPARITYMAPPING_STRING_SIZE]={0};
char disparityCalibrationOutputPath[DISPARITYMAPPING_STRING_SIZE]={0};

unsigned int swapColorFeeds=0;
unsigned int SADWindowSize = 45;
unsigned int shiftYLeft=0;
unsigned int shiftYRight=0;
unsigned int speckleRange=32;


unsigned int performCalibration=0;
float calibSquareSize=0.0;
unsigned int horizontalSquares=0,verticalSquares=0;


int initArgs_DisparityMapping(int argc, char *argv[])
{
  fprintf(stderr,GREEN "\nDisparity Mapping Processor now parsing initialization parameters\n" NORMAL);
  fprintf(stderr,GREEN "_________________________________________________________________\n" NORMAL);

  int i=0;
  for (i=0; i<argc; i++)
  {
     if (strcmp(argv[i],"-disparityUseCalibration")==0)
        {
         disparityCalibrationUsed=1;
         snprintf(disparityCalibrationPath,DISPARITYMAPPING_STRING_SIZE,"%s",argv[i+1]);
         fprintf(stderr,GREEN "Disparity Mapping %s \n" NORMAL,disparityCalibrationPath);
         newKindOfDisplayCalibrationReading(disparityCalibrationPath);
         //oldKindOfDisplayCalibrationReading(disparityCalibrationPath);
        } else
     if (strcmp(argv[i],"-disparitySwapColorFeeds")==0)
        {
         swapColorFeeds=1;
         fprintf(stderr,GREEN "Disparity Mapping will swap L/R color feeds \n" NORMAL);
        } else
     if (strcmp(argv[i],"-disparitySADWindowSize")==0)
        {
         SADWindowSize=atoi(argv[i+1]);
         fprintf(stderr,GREEN "SADWindowSize set to %u \n" NORMAL,SADWindowSize);
        } else
     if (strcmp(argv[i],"-disparityshiftYLeft")==0)
        {
         shiftYRight=atoi(argv[i+1]);
         fprintf(stderr,RED "shiftYRight set to %u \n" NORMAL,shiftYRight);
        } else
     if (strcmp(argv[i],"-disparityshiftYRight")==0)
        {
         shiftYLeft=atoi(argv[i+1]);
         fprintf(stderr,RED "shiftYLeft set to %u \n" NORMAL,shiftYLeft);
        } else
     if (strcmp(argv[i],"-disparitySpeckleRange")==0)
        {
         speckleRange=atoi(argv[i+1]);
         fprintf(stderr,GREEN "speckleRange set to %u \n" NORMAL,speckleRange);
        } else
     if (strcmp(argv[i],"-disparityHelp")==0)
        {
         fprintf(stderr,RED "TODO:Add help message here \n" NORMAL);
        } else
     if (strcmp(argv[i],"-disparityCalibrate")==0)
        {
         fprintf(stderr,RED "Will do calibration on disparity input\n" NORMAL);
         performCalibration=1;
         horizontalSquares=atoi(argv[i+1]);
         verticalSquares=atoi(argv[i+2]);
         calibSquareSize=atof(argv[i+3]);
         snprintf(disparityCalibrationOutputPath,DISPARITYMAPPING_STRING_SIZE,"%s",argv[i+4]);
         //
        }


  }

 fprintf(stderr,GREEN "_________________________________________________________________\n\n" NORMAL);
 return 1;
}



int setConfigStr_DisparityMapping(char * label,char * value)
{
 return 0;
}

int setConfigInt_DisparityMapping(char * label,int value)
{
return 0;
}



unsigned char * getDataOutput_DisparityMapping(unsigned int stream , unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
 return 0;
}



int addDataInput_DisparityMapping(unsigned int stream , void * data, unsigned int width, unsigned int height,unsigned int channels,unsigned int bitsperpixel)
{
  if (stream==0)
  {
    unsigned int colorFrameSize = width*height*channels*(bitsperpixel/8);
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
    unsigned int depthFrameSize = width*height*channels*(bitsperpixel/8);
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




unsigned short * getDepth_DisparityMapping(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
    *width=depthWidth; *height=depthHeight; *channels=depthChannels;  *bitsperpixel=depthBitsperpixel;
    return depthFrame;
}


unsigned char * getColor_DisparityMapping(unsigned int * width, unsigned int * height,unsigned int * channels,unsigned int * bitsperpixel)
{
    *width=colorWidth; *height=colorHeight; *channels=colorChannels;  *bitsperpixel=colorBitsperpixel;
    return colorFrame;
}




int processData_DisparityMapping()
{
    int retres=0;
    // Start and end times
    time_t startTime , endTime;
    time(&startTime);

    unsigned char * colorPTR = colorFrame ;

    unsigned int x,y;
    for (y=0; y<colorHeight; y++)
    {
     for (x=0; x<colorWidth*colorChannels; x++)
     {
       //*colorPTR=(unsigned char) rand()%255;
       ++colorPTR;
     }
    }

   passNewFrame(colorFrame,colorWidth,colorHeight,swapColorFeeds,shiftYLeft,shiftYRight);


   if (performCalibration)
   {
    fprintf(stderr,"doing calibration\n");
    return doCalibrationStep(&leftRGB , &rightRGB , &greyLeft , &greyRight ,horizontalSquares,verticalSquares,calibSquareSize , disparityCalibrationOutputPath);
   } else
   {
    fprintf(stderr,"doing regular loop\n");

    #if UNIFY_LKFLOWS
     doStereoLKOpticalFlow(
                            leftRGB,greyLeft,greyLastLeft ,
                            rightRGB,greyRight,greyLastRight
                          );
    #else
     doLKOpticalFlow(leftRGB,greyLastLeft,greyLeft);
     doLKOpticalFlow(rightRGB,greyLastRight,greyRight);
    #endif // UNIFY_LKFLOWS

    doSGBM( &leftRGB, &rightRGB , SADWindowSize ,  speckleRange, disparityCalibrationPath );
    retres=1;
   }



 time(&endTime);

 // Time elapsed
 double seconds = difftime (endTime, startTime);

 if (seconds == 0.0 ) { seconds = 0.0001; }
 fprintf(stderr,"DisparityMapping Node Achieving %0.2f fps \n",(float) 1/seconds);


 return retres;
}



int cleanup_DisparityMapping()
{
    if (colorFrame!=0) { free(colorFrame); colorFrame=0; }
    if (depthFrame!=0) { free(depthFrame); depthFrame=0; }
    return 1;
}



int stop_DisparityMapping()
{

  fprintf(stderr,GREEN "\nDisparity Mapping Processor now gracefully stopping\n" NORMAL);
  fprintf(stderr,GREEN "_________________________________________________________________\n" NORMAL);
    cleanup_DisparityMapping();
    if (performCalibration)
    {
     finalizeCalibration(disparityCalibrationOutputPath,horizontalSquares,verticalSquares,calibSquareSize );
    }
  fprintf(stderr,GREEN "_________________________________________________________________\n" NORMAL);
    return 1;
}


