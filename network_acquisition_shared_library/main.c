#include <stdio.h>
#include <stdlib.h>

#include "NetworkAcquisition.h"


#include "../acquisition/Acquisition.h"

#include <string.h>
#include <math.h>

#define MAX_NETWORK_DEVICES 5

#define PRINT_COMMENTS 1
#define PRINT_DEBUG_EACH_CALL 0

#define DEFAULT_FOCAL_LENGTH 120.0
#define DEFAULT_PIXEL_SIZE 0.1052

struct NetworkVirtualDevice
{
 unsigned int cycle;
 unsigned int templateWIDTH;
 unsigned int templateHEIGHT;
 unsigned long lastColorTimestamp;
 char * templateColorFrame;
 unsigned long lastDepthTimestamp;
 short * templateDepthFrame;

 struct calibration calibRGB;
 struct calibration calibDepth;

};

struct NetworkVirtualDevice device[MAX_NETWORK_DEVICES]={0};

#define BUILD_NETWORK 1

#if BUILD_NETWORK

double getNetworkColorPixelSize(int devID)   { return DEFAULT_PIXEL_SIZE; }
double getNetworkColorFocalLength(int devID)
{
   return DEFAULT_FOCAL_LENGTH;
}

double getNetworkDepthFocalLength(int devID)
{  return DEFAULT_FOCAL_LENGTH;
}
double getNetworkDepthPixelSize(int devID) { return DEFAULT_PIXEL_SIZE; }



int getNetworkColorCalibration(int devID,struct calibration * calib)
{
    return 1;
}

int getNetworkDepthCalibration(int devID,struct calibration * calib)
{
    return 1;
}


int setNetworkColorCalibration(int devID,struct calibration * calib)
{
    return 1;
}

int setNetworkDepthCalibration(int devID,struct calibration * calib)
{
    return 1;
}





int startNetworkModule(unsigned int max_devs,char * settings)
{
    return 1;
}




int getNetworkNumberOfDevices() { return 1; }

int stopNetworkModule()
{
   return 1;
}

int createNetworkDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
  return 0;
}



int destroyNetworkDevice(int devID)
{
  return 1;
}


int seekNetworkFrame(int devID,unsigned int seekFrame)
{
  return 1;
}

int snapNetworkFrames(int devID)
{

  return 1;
}

//Color Frame getters
unsigned long getLastNetworkColorTimestamp(int devID) { return device[devID].lastColorTimestamp; }
int getNetworkColorWidth(int devID)        { return device[devID].templateWIDTH; }
int getNetworkColorHeight(int devID)       { return device[devID].templateHEIGHT; }
int getNetworkColorDataSize(int devID)     { return device[devID].templateHEIGHT*device[devID].templateWIDTH * 3; }
int getNetworkColorChannels(int devID)     { return 3; }
int getNetworkColorBitsPerPixel(int devID) { return 8; }

// Frame Grabber should call this function for color frames
char * getNetworkColorPixels(int devID)    { return device[devID].templateColorFrame; }




   //Depth Frame getters
unsigned long getLastNetworkDepthTimestamp(int devID) { return device[devID].lastDepthTimestamp; }
int getNetworkDepthWidth(int devID)    { return device[devID].templateWIDTH; }
int getNetworkDepthHeight(int devID)   { return device[devID].templateHEIGHT; }
int getNetworkDepthDataSize(int devID) { return device[devID].templateWIDTH*device[devID].templateHEIGHT; }
int getNetworkDepthChannels(int devID)     { return 1; }
int getNetworkDepthBitsPerPixel(int devID) { return 16; }

// Frame Grabber should call this function for depth frames
char * getNetworkDepthPixels(int devID) { return (char *) device[devID].templateDepthFrame; }

#else
//Null build
int startNetworkModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startNetworkModule called on a dummy build of NetworkAcquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_NETWORK 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
