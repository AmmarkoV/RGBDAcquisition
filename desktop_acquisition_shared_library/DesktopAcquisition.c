#include <stdio.h>
#include <stdlib.h>

#include "../acquisition/Acquisition.h"
#include "../tools/Primitives/modules.h"
#include "../3dparty/xwd-1.0.5/XwdLib.h"

#include "DesktopAcquisition.h"


#include <string.h>
#include <math.h>


#define MAX_TEMPLATE_DEVICES 1


struct DesktopVirtualDevice
{
 char intialized;

 unsigned int cycle;
 unsigned int totalFrames;
 unsigned int safeGUARD;

 unsigned int frameRate;
 unsigned int colorWidth;
 unsigned int colorHeight;
 unsigned long lastColorTimestamp;
 unsigned char * colorFrame;

 unsigned int disableRGBStream;
};

struct DesktopVirtualDevice device[MAX_TEMPLATE_DEVICES]={0};




int getDesktopCapabilities(int devID,int capToAskFor)
{
  switch (capToAskFor)
  {
    case CAP_VERSION :            return CAP_ENUM_LIST_VERSION;  break;
    case CAP_LIVESTREAM :         return 1;                      break;
    case CAP_PROVIDES_LOCATIONS : return 0;                      break;
  };
 return 0;
}




int mapDesktopDepthToRGB(int devID) { return 0; }
int mapDesktopRGBToDepth(int devID) { return 0; }
int switchDesktopToColorStream(int devID) { return 1; }

int getDesktopNumberOfColorStreams(int devID) { return 1; /*TODO support multiple streams*/ }

double getDesktopColorPixelSize(int devID)   { return 1; }
double getDesktopColorFocalLength(int devID) { return 1; }

double getDesktopDepthFocalLength(int devID)  { return 1; }
double getDesktopDepthPixelSize(int devID)    { return 1; }



int enableDesktopStream(int devID,unsigned int streamID)
{
    if (streamID==0) { device[devID].disableRGBStream=0; } else
    if (streamID==1) { return 0;  }
    return 1;
}

int disableDesktopStream(int devID,unsigned int streamID)
{
    if (streamID==0) { device[devID].disableRGBStream=1; } else
    if (streamID==1) { return 1; }
    return 1;
}



int startDesktopModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr," initXwdLib : ");
    initXwdLib(0,0);
    fprintf(stderr," ok \n");
    return 1;
}

int deviceIsSafeToUse(int devID)
{
 return (devID==0);
}


int getDesktopNumberOfDevices() { return 1; }

int stopDesktopModule()
{
   closeXwdLib();

   unsigned int devID = 0;
   for (devID=0; devID<MAX_TEMPLATE_DEVICES; devID++)
   {
     if (device[devID].colorFrame!=0) { free(device[devID].colorFrame); device[devID].colorFrame=0; }
   }

   return 1;
}


int listDesktopDevices(int devID,char * output, unsigned int maxOutput)
{
    return 0;
}


int createDesktopDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
  fprintf(stderr,"Be careful it is important you use -noDepth and the correct -resolution 1920 1080\n");
  fprintf(stderr," allocating size ( %ux%u @ %u fps ) : ",width,height,framerate);
  device[devID].colorWidth = width;
  device[devID].colorHeight = height;
  device[devID].frameRate = framerate;

  device[devID].colorFrame = (unsigned char* ) malloc(sizeof(char) * 3 * width * height);

  if (device[devID].colorFrame!=0) { device[devID].intialized=1; }
  fprintf(stderr," ok \n");

  return device[devID].intialized;
}



int destroyDesktopDevice(int devID)
{
  if (device[devID].colorFrame!=0) { free(device[devID].colorFrame); device[devID].colorFrame=0; }
  return 1;
}

int getTotalDesktopFrameNumber(int devID)
{
  return device[devID].totalFrames;
}

int getCurrentDesktopFrameNumber(int devID)
{
  return device[devID].cycle;
}


int snapDesktopFrames(int devID)
{
  if (!device[devID].intialized) { return 0; }

  getScreen(
            device[devID].colorFrame ,
            &device[devID].colorWidth,
            &device[devID].colorHeight
           );

  ++device[devID].cycle;
  ++device[devID].totalFrames;
  return 1;
}


int controlDesktopFlow(int devID,float newFlowState)
{
  device[devID].cycle = newFlowState;
  return 0;
}


int seekRelativeDesktopFrame(int devID,signed int seekFrame) { return 0; }
int seekDesktopFrame(int devID,unsigned int seekFrame)       { return 0; }


//Color Frame getters
unsigned long getLastDesktopColorTimestamp(int devID) { return device[devID].lastColorTimestamp; }
int getDesktopColorWidth(int devID)        { return device[devID].colorWidth; }
int getDesktopColorHeight(int devID)       { return device[devID].colorHeight; }
int getDesktopColorDataSize(int devID)     { return device[devID].colorHeight*device[devID].colorWidth * 3; }
int getDesktopColorChannels(int devID)     { return 3; }
int getDesktopColorBitsPerPixel(int devID) { return 8; }

// Frame Grabber should call this function for color frames
unsigned char * getDesktopColorPixels(int devID)    { return device[devID].colorFrame; }


   //Depth Frame getters
unsigned long getLastDesktopDepthTimestamp(int devID) { return 0; }
int getDesktopDepthWidth(int devID)    { return 0; }
int getDesktopDepthHeight(int devID)   { return 0; }
int getDesktopDepthDataSize(int devID) { return 0; }
int getDesktopDepthChannels(int devID)     { return 0; }
int getDesktopDepthBitsPerPixel(int devID) { return 0; }

// Frame Grabber should call this function for depth frames
char * getDesktopDepthPixels(int devID) { return (char *) 0; }
char * getDesktopDepthPixelsFlipped(int devID) { return (char *) 0; }

