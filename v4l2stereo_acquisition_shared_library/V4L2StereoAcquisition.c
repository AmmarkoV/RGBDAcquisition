#include <stdio.h>
#include <stdlib.h>
#include "V4L2StereoAcquisition.h"

#include "../tools/Primitives/modules.h"

#if BUILD_V4L2

#define MAX_DEVICES 10

#include "../acquisition/Acquisition.h"
#include "../v4l2_acquisition_shared_library/V4L2Acquisition.h"
#include <string.h>

#define MEMPLACE3(x,y,width) ( y * ( width * 3 ) + x*3 )
struct v4l2StereoDevices
{
    unsigned char * bothImage;
    unsigned int bothWidth;
    unsigned int bothHeight;

    unsigned int activeStream;

    unsigned char * leftFrame;
    unsigned char * rightFrame;
    unsigned int sizeOfFeedX,sizeOfFeedY;
    unsigned int sizePerFrame;
};

struct v4l2StereoDevices devices[MAX_DEVICES]={0};




int getV4L2StereoCapabilities(int devID,int capToAskFor)
{
  switch (capToAskFor)
  {
    case CAP_VERSION :            return CAP_ENUM_LIST_VERSION;  break;
    case CAP_LIVESTREAM :         return 1;                      break;
    case CAP_PROVIDES_LOCATIONS : return 0;                      break;
  };
 return 0;
}



int startV4L2StereoModule(unsigned int max_devs,char * settings)
{
 return startV4L2Module(max_devs,settings);
}

int getV4L2Stereo()
{
 return 0;
} // This has to be called AFTER startV4L2Stereo

int stopV4L2StereoModule()
{
 return stopV4L2Module();
}


int getV4L2StereoNumberOfDevices()
{
    return 1;
}

int getDevIDForV4L2StereoName(char * devName)
{
 return 0;
}

   //Basic Per Device Operations
int createV4L2StereoDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
 devices[devID].activeStream=2;

 char devName1[512]={0};
 char devName2[512]={0};
 char * secondString = strstr(devName,",");
 if ( (secondString==0) || (secondString==devName) )
    {
      fprintf(stderr,"Could not find two strings seperated by comma for stereo device names , using default");
      return 0;
    }

 strncpy(devName1, devName , secondString-devName);
 strncpy(devName2, secondString+1 , strlen(secondString)-1 ) ;

 fprintf(stderr,"Creating virtual stereo acquisitioin device using %s and %s \n",devName1,devName2);

 unsigned int retres=0;
 retres+=createV4L2Device(devID+0,devName1,width,height,framerate);
 retres+=3*createV4L2Device(devID+1,devName2,width,height,framerate);


 snapV4L2Frames(devID+0);
 snapV4L2Frames(devID+1);
 snapV4L2Frames(devID+0); //Snap another frame to achieve sync

 unsigned int actWidth =  getV4L2ColorWidth(devID+0);
 unsigned int actHeight =  getV4L2ColorHeight(devID);
 if (actWidth < getV4L2ColorWidth(devID+1) ) { actWidth=getV4L2ColorWidth(devID+1); }
 if (actHeight < getV4L2ColorHeight(devID+1) ) { actHeight=getV4L2ColorHeight(devID+1); }

 devices[devID].leftFrame = (char*) malloc (actWidth *actHeight * 3 * sizeof(char));
 devices[devID].rightFrame= (char*) malloc (actWidth *actHeight * 3 * sizeof(char));
 devices[devID].bothImage=  (char*) malloc (actWidth *actHeight * 3 * 2 * sizeof(char));
 devices[devID].sizeOfFeedX=actWidth;
 devices[devID].sizeOfFeedY=actHeight;
 devices[devID].sizePerFrame = actWidth * actHeight * 3 ;

 if (retres==0)
 {
   fprintf(stderr,"Could not initialize any of the cameras!\n");
   return 0;
 } else
 if (retres<4)
 {
   fprintf(stderr,"Could not initialize both of the cameras!\n");
   if ( retres == 3 )  { fprintf(stderr,"Destroying successfully initialized device"); destroyV4L2Device(devID+1); } else
                       { fprintf(stderr,"V4L2 device %s failed to be initialized!\n",devName2); }
   if ( retres == 1 )  { fprintf(stderr,"Destroying successfully initialized device"); destroyV4L2Device(devID+0);  } else
                       { fprintf(stderr,"V4L2 device %s failed to be initialized!\n",devName1); }
   return 0;
 }

 return 1;
}

int destroyV4L2StereoDevice(int devID)
{
 destroyV4L2Device(devID+0);
 destroyV4L2Device(devID+1);


 if ( devices[devID].bothImage != 0 )  { free(devices[devID].bothImage); devices[devID].bothImage=0; }
 if ( devices[devID].leftFrame != 0 )  { free(devices[devID].leftFrame);  devices[devID].leftFrame=0; }
 if ( devices[devID].rightFrame != 0 ) { free(devices[devID].rightFrame); devices[devID].rightFrame=0; }

 return 0;
}

int seekV4L2StereoFrame(int devID,unsigned int seekFrame)
{
 return 0;
}





int bitbltRGB(unsigned char * target,  unsigned int tX,  unsigned int tY , unsigned int targetWidth , unsigned int targetHeight ,
              unsigned char * source , unsigned int sX, unsigned int sY  , unsigned int sourceWidth , unsigned int sourceHeight ,
              unsigned int width , unsigned int height)
{
  if ( (target==0)||(source==0) ) { return 0; }
  if ( (width==0)&&(height==0) ) { return 0; }
  if ( (sourceWidth==0)&&(sourceHeight==0) ) { return 0; }

  fprintf(stderr,"BitBlt an area of target image %u,%u  that starts at %u,%u \n",tX,tY,targetWidth,targetHeight);
  fprintf(stderr,"BitBlt an area of source image %u,%u  that starts at %u,%u \n",sX,sY,sourceWidth,sourceHeight);
  fprintf(stderr,"BitBlt size was width %u height %u \n",width,height);
  //Check for bounds -----------------------------------------
  if (tX+width>=targetWidth) { width=targetWidth-tX-1;  }
  if (tY+height>=targetHeight) { height=targetHeight-tY-1;  }

  if (sX+width>=sourceWidth) { width=sourceWidth-sX-1;  }
  if (sY+height>=sourceHeight) { height=sourceHeight-sY-1;  }
  //----------------------------------------------------------
  fprintf(stderr,"BitBlt size NOW is width %u height %u \n",width,height);


  unsigned char * sourcePTR; unsigned char * sourceLineLimitPTR; unsigned char * sourceLimitPTR; unsigned int sourceLineSkip;
  unsigned char * targetPTR; /*unsigned char * targetLimitPTR;*/  unsigned int targetLineSkip;


  sourcePTR      = source+ MEMPLACE3(sX,sY,sourceWidth);
  sourceLimitPTR = source+ MEMPLACE3((sX+width),(sY+height),sourceWidth);
  sourceLineSkip = (sourceWidth-width) * 3;
  sourceLineLimitPTR = sourcePTR + (width*3);
  fprintf(stderr,"SOURCE (RGB %u/%u)  Starts at %u,%u and ends at %u,%u\n",sourceWidth,sourceHeight,sX,sY,sX+width,sY+height);
  fprintf(stderr,"sourcePTR is %p\n",sourcePTR);
  fprintf(stderr,"sourceLimitPTR is %p\n",(void*) sourceLimitPTR);
  fprintf(stderr,"sourceLineSkip is %u\n",        sourceLineSkip);
  fprintf(stderr,"sourceLineLimitPTR is %p\n",sourceLineLimitPTR);


  targetPTR      = target + MEMPLACE3(tX,tY,targetWidth);
  //targetLimitPTR = target + MEMPLACE3((tX+width),(tY+height),targetWidth);
  targetLineSkip = (targetWidth-width) * 3;
  fprintf(stderr,"targetPTR is %p\n",targetPTR);
  fprintf(stderr,"targetLineSkip is %u\n", targetLineSkip);
  fprintf(stderr,"TARGET (RGB %u/%u)  Starts at %u,%u and ends at %u,%u\n",targetWidth,targetHeight,tX,tY,tX+width,tY+height);

  while (sourcePTR < sourceLimitPTR)
  {
     while (sourcePTR < sourceLineLimitPTR)
     {
        //fprintf(stderr,"Reading Triplet sourcePTR %p targetPTR is %p\n",sourcePTR  ,targetPTR);
        *targetPTR = *sourcePTR; ++targetPTR; ++sourcePTR;
        *targetPTR = *sourcePTR; ++targetPTR; ++sourcePTR;
        *targetPTR = *sourcePTR; ++targetPTR; ++sourcePTR;
     }

    sourceLineLimitPTR+= sourceWidth*3;//*3;
    targetPTR+=targetLineSkip;
    sourcePTR+=sourceLineSkip;
  }
 return 1;
}



int snapV4L2StereoFrames(int devID)
{
 snapV4L2Frames(devID+0);
 snapV4L2Frames(devID+1);
 memcpy( devices[devID].leftFrame , getV4L2ColorPixels(devID+0) , devices[devID].sizePerFrame );
 memcpy( devices[devID].rightFrame , getV4L2ColorPixels(devID+1) , devices[devID].sizePerFrame );

 return 0;
}


int getV4L2StereoNumberOfColorStreams(int devID) { return 2; }
int switchV4L2StereoToColorStream(int devID,unsigned int streamToActivate) { devices[devID].activeStream=streamToActivate; return 1; }

//Color Frame getters
int getV4L2StereoColorWidth(int devID) { return getV4L2ColorWidth(devID)*2; }
int getV4L2StereoColorHeight(int devID) { return getV4L2ColorHeight(devID); }
int getV4L2StereoColorDataSize(int devID) { return getV4L2ColorDataSize(devID); }
int getV4L2StereoColorChannels(int devID) {  return getV4L2ColorChannels(devID); }
int getV4L2StereoColorBitsPerPixel(int devID) {  return getV4L2ColorBitsPerPixel(devID); }



unsigned char * getV4L2StereoColorPixels(int devID)
{
  if (devices[devID].activeStream<2)
  {
      return getV4L2ColorPixels(devID+devices[devID].activeStream);
  }


  unsigned int width =  getV4L2ColorWidth(devID);
  unsigned int height =  getV4L2ColorHeight(devID);
  //unsigned int channels =  getV4L2ColorChannels(devID);

  // This is not needed if (devices[devID].bothImage != 0 ) { free(devices[devID].bothImage); devices[devID].bothImage=0;  }

  if (devices[devID].bothImage == 0 )
  {
     devices[devID].bothImage = ( unsigned char * ) malloc (sizeof(unsigned char) * 2 * getV4L2ColorWidth(devID) * getV4L2ColorHeight(devID) * getV4L2ColorChannels(devID) );
  }

  bitbltRGB(devices[devID].bothImage,0,0,width*2,height,
            getV4L2ColorPixels(devID+0),0,0,width,height,
            width,height);

  bitbltRGB(devices[devID].bothImage,width,0,width*2,height,
            getV4L2ColorPixels(devID+1),0,0,width,height,
            width,height);

 return devices[devID].bothImage;

}

unsigned char * getV4L2StereoColorPixelsLeft(int devID) {  return getV4L2ColorPixels(devID+0); }
unsigned char * getV4L2StereoColorPixelsRight(int devID) {  return getV4L2ColorPixels(devID+1); }

double getV4L2StereoColorFocalLength(int devID)
{
 return 0;
}

double getV4L2StereoColorPixelSize(int devID)
{
 return 0;
}

   //Depth Frame getters
int getV4L2StereoDepthWidth(int devID) { return 0; }
int getV4L2StereoDepthHeight(int devID) { return 0; }
int getV4L2StereoDepthDataSize(int devID) { return 0; }
int getV4L2StereoDepthChannels(int devID) { return 0; }
int getV4L2StereoDepthBitsPerPixel(int devID) { return 0; }
char * getV4L2StereoDepthPixels(int devID) {  return 0; }
double getV4L2StereoDepthFocalLength(int devID) { return 0; }
double getV4L2StereoDepthPixelSize(int devID) { return 0; }
#else
//Null build
int start4L2StereoModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"start4L2StereoModule called on a dummy build of V4L2StereoAcquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_V4L2 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
