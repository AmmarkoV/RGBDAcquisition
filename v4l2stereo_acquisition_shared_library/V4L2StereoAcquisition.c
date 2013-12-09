#include <stdio.h>
#include <stdlib.h>
#include "V4L2StereoAcquisition.h"

#if BUILD_V4L2
#include "../acquisition/Acquisition.h"
#include "../v4l2_acquisition_shared_library/V4L2Acquisition.h"
#include <string.h>

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
 return 0;
}

int seekV4L2StereoFrame(int devID,unsigned int seekFrame)
{
 return 0;
}

int snapV4L2StereoFrames(int devID)
{
 snapV4L2Frames(devID+0);
 snapV4L2Frames(devID+1);
 return 0;
}

//Color Frame getters
int getV4L2StereoColorWidth(int devID) { return getV4L2ColorWidth(devID); }
int getV4L2StereoColorHeight(int devID) { return getV4L2ColorHeight(devID); }
int getV4L2StereoColorDataSize(int devID) { return getV4L2ColorDataSize(devID); }
int getV4L2StereoColorChannels(int devID) {  return getV4L2ColorChannels(devID); }
int getV4L2StereoColorBitsPerPixel(int devID) {  return getV4L2ColorBitsPerPixel(devID); }

unsigned char * getV4L2StereoColorPixels(int devID) { return getV4L2ColorPixels(devID+0); }
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
