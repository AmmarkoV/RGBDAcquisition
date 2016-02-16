#include <stdio.h>
#include <stdlib.h>

#include "DepthSenseAcquisition.h"

#define BUILD_DEPTHSENSE 1

#if BUILD_DEPTHSENSE
#include "../acquisition/Acquisition.h"

#include "../tools/Primitives/modules.h"

#include <string.h>
#include <math.h>

#include "templateAcquisitionHelper.h"

#define SAFEGUARD_VALUE 123123
#define MAX_TEMPLATE_DEVICES 5
#define MAX_LINE_CALIBRATION 1024

#define DEFAULT_FOCAL_LENGTH 120.0
#define DEFAULT_PIXEL_SIZE 0.1052

#define REALLOCATE_ON_EVERY_SNAP 0
 #if REALLOCATE_ON_EVERY_SNAP
         #warning "Reallocating arrays on every frame ,  performance will be impacted by this "
        #else
         #warning "Not Reallocating arrays on every frame this means better performance but if stream dynamically changes resolution from frame to frame this could be a problem"
 #endif




struct DepthSenseVirtualDevice
{
 char intialized;

 struct calibration calibRGB;
 unsigned int disableRGBStream;
 struct calibration calibDepth;
 unsigned int disableDepthStream;

};

struct DepthSenseVirtualDevice device[MAX_TEMPLATE_DEVICES]={0};




int getDepthSenseCapabilities(int devID,int capToAskFor)
{
  switch (capToAskFor)
  {
    case CAP_VERSION :            return CAP_ENUM_LIST_VERSION;  break;
    case CAP_LIVESTREAM :         return 1;                      break;
    case CAP_PROVIDES_LOCATIONS : return 1; fprintf(stderr,"TODO : Provide locations from DepthSenses"); break;
  };
 return 0;
}




int mapDepthSenseDepthToRGB(int devID) { return 0; }
int mapDepthSenseRGBToDepth(int devID) { return 0; }
int switchDepthSenseToColorStream(int devID) { return 1; }

int getDepthSenseNumberOfColorStreams(int devID) { return 1; /*TODO support multiple streams*/ }

double getDepthSenseColorPixelSize(int devID)   { return DEFAULT_PIXEL_SIZE; }
double getDepthSenseColorFocalLength(int devID)
{
  return DEFAULT_FOCAL_LENGTH;
}

double getDepthSenseDepthFocalLength(int devID)
{
    return DEFAULT_FOCAL_LENGTH;
}
double getDepthSenseDepthPixelSize(int devID) { return DEFAULT_PIXEL_SIZE; }


#if USE_CALIBRATION
int getDepthSenseColorCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &device[devID].calibRGB,sizeof(struct calibration));
    return 1;
}

int getDepthSenseDepthCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &device[devID].calibDepth,sizeof(struct calibration));
    return 1;
}


int setDepthSenseColorCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) &device[devID].calibRGB , (void*) calib,sizeof(struct calibration));
    return 1;
}

int setDepthSenseDepthCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) &device[devID].calibDepth , (void*) calib,sizeof(struct calibration));
    return 1;
}
#endif



int enableDepthSenseStream(int devID,unsigned int streamID)
{
 return 0;
}

int disableDepthSenseStream(int devID,unsigned int streamID)
{
 return 0;
}



int startDepthSenseModule(unsigned int max_devs,char * settings)
{
 return 0;
}

int deviceIsSafeToUse(int devID)
{
 return 0;
}


int getDepthSenseNumberOfDevices() { return 1; }

int stopDepthSenseModule()
{
 return 0;
}


int listDepthSenseDevices(int devID,char * output, unsigned int maxOutput)
{
 return 0;
}


int createDepthSenseDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
 return 0;
}



int destroyDepthSenseDevice(int devID)
{
  return 1;
}

int getTotalDepthSenseFrameNumber(int devID)
{
    return 0;
}

int getCurrentDepthSenseFrameNumber(int devID)
{
  return 0;
}


int snapDepthSenseFrames(int devID)
{
  return 0;
}


int controlDepthSenseFlow(int devID,float newFlowState)
{
 return 0;
}


int seekRelativeDepthSenseFrame(int devID,signed int seekFrame)
{
 return 0;
}

int seekDepthSenseFrame(int devID,unsigned int seekFrame)
{
 return 0;
}


//Color Frame getters
unsigned long getLastDepthSenseColorTimestamp(int devID) { return device[devID].lastColorTimestamp; }
int getDepthSenseColorWidth(int devID)        { return device[devID].templateColorWidth; }
int getDepthSenseColorHeight(int devID)       { return device[devID].templateColorHeight; }
int getDepthSenseColorDataSize(int devID)     { return device[devID].templateColorHeight*device[devID].templateColorWidth * 3; }
int getDepthSenseColorChannels(int devID)     { return 3; }
int getDepthSenseColorBitsPerPixel(int devID) { return 8; }

// Frame Grabber should call this function for color frames
unsigned char * getDepthSenseColorPixels(int devID)    { return device[devID].templateColorFrame; }




   //Depth Frame getters
unsigned long getLastDepthSenseDepthTimestamp(int devID) { return device[devID].lastDepthTimestamp; }
int getDepthSenseDepthWidth(int devID)    { return device[devID].templateDepthWidth; }
int getDepthSenseDepthHeight(int devID)   { return device[devID].templateDepthHeight; }
int getDepthSenseDepthDataSize(int devID) { return device[devID].templateDepthWidth*device[devID].templateDepthHeight; }
int getDepthSenseDepthChannels(int devID)     { return 1; }
int getDepthSenseDepthBitsPerPixel(int devID) { return 16; }

// Frame Grabber should call this function for depth frames
char * getDepthSenseDepthPixels(int devID) { return (char *) device[devID].templateDepthFrame; }

char * getDepthSenseDepthPixelsFlipped(int devID) {
                                                  flipDepth(device[devID].templateDepthFrame,device[devID].templateDepthWidth, device[devID].templateDepthHeight);
                                                  return (char *) device[devID].templateDepthFrame;
                                                }

#else
//Null build
int startDepthSenseModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startDepthSenseModule called on a dummy build of DepthSenseAcquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_TEMPLATE 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
