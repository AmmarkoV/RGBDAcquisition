#include <stdio.h>
#include <stdlib.h>

#include "DepthSenseAcquisition.hxx"

#define BUILD_DEPTHSENSE 1

#if BUILD_DEPTHSENSE
#include "../acquisition/Acquisition.h"

#include "../tools/Primitives/modules.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//BOOST CRAP HERE
#include <time.h>
#include <vector>
#include <exception>

//THIS LIBRARY IS BASED ON PH4MS DepthSense grabber
#include "../3dparty/DepthSenseGrabber/DepthSenseGrabberCore/DepthSenseGrabberCore.hxx"

#define MAX_DEPTHSENSE_DEVICES 5
#define MAX_LINE_CALIBRATION 1024

#define DEFAULT_FOCAL_LENGTH 120.0
#define DEFAULT_PIXEL_SIZE 0.1052



struct DepthSenseVirtualDevice
{
 char intialized;

 struct calibration calibRGB;
 unsigned int disableRGBStream;
 struct calibration calibDepth;
 unsigned int disableDepthStream;


 char interpolateDepthFlag,saveColorAcqFlag,saveDepthAcqFlag,saveColorSyncFlag,saveDepthSyncFlag,saveConfidenceFlag;

 char buildColorSyncFlag,buildDepthSyncFlag,buildConfidenceFlag;

 int flagColorFormat; // VGA, WXGA or NHD

 int widthColor, heightColor , widthDepthAcq ,heightDepthAcq;

 unsigned char * pixelsColorSync;
 unsigned char * pixelsColorAcq;

 unsigned short * pixelsDepthAcq;
 unsigned short * pixelsDepthSync;
 unsigned short * pixelsConfidenceQVGA;


 int frameCount;
 int timeStamp;

};

struct DepthSenseVirtualDevice device[MAX_DEPTHSENSE_DEVICES]={0};




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

int getDepthSenseNumberOfColorStreams(int devID) { return 1; }

double getDepthSenseColorPixelSize(int devID)   { return DEFAULT_PIXEL_SIZE; }
double getDepthSenseColorFocalLength(int devID) { return DEFAULT_FOCAL_LENGTH; }

double getDepthSenseDepthFocalLength(int devID) { return DEFAULT_FOCAL_LENGTH; }
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
 fprintf(stderr,"Starting depthsense module doesnt do much.. \n");
 return 1;
}

int deviceIsSafeToUse(int devID)
{
 return 0;
}


int getDepthSenseNumberOfDevices() { return 1; }

int stopDepthSenseModule()
{
 return 1;
}


int listDepthSenseDevices(int devID,char * output, unsigned int maxOutput)
{
 return 0;
}


int createDepthSenseDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
   device[devID].flagColorFormat = FORMAT_VGA_ID; // VGA, WXGA or NHD
   device[devID].interpolateDepthFlag = 1;

   switch (device[devID].flagColorFormat)
    {
        case FORMAT_VGA_ID:
              device[devID].widthColor = FORMAT_VGA_WIDTH;
              device[devID].heightColor = FORMAT_VGA_HEIGHT;
            break;
        case FORMAT_WXGA_ID:
              device[devID].widthColor = FORMAT_WXGA_WIDTH;
              device[devID].heightColor = FORMAT_WXGA_HEIGHT;
            break;
        case FORMAT_NHD_ID:
              device[devID].widthColor = FORMAT_NHD_WIDTH;
              device[devID].heightColor = FORMAT_NHD_HEIGHT;
            break;
    };

if (device[devID].interpolateDepthFlag)
    {
        device[devID].widthDepthAcq = FORMAT_VGA_WIDTH;
        device[devID].heightDepthAcq = FORMAT_VGA_HEIGHT;
    } else
    {
        device[devID].widthDepthAcq = FORMAT_QVGA_WIDTH;
        device[devID].heightDepthAcq = FORMAT_QVGA_HEIGHT;
    }




 device[devID].saveColorAcqFlag   = 1;
 device[devID].saveDepthAcqFlag   = 1;
 device[devID].saveColorSyncFlag  = 1;
 device[devID].saveDepthSyncFlag  = 1;
 device[devID].saveConfidenceFlag = 1;

 device[devID].buildColorSyncFlag = device[devID].saveColorSyncFlag;
 device[devID].buildDepthSyncFlag = device[devID].saveDepthSyncFlag;
 device[devID].buildConfidenceFlag = device[devID].saveConfidenceFlag;


   start_capture(
                        device[devID].flagColorFormat,
                        device[devID].interpolateDepthFlag,
                        device[devID].buildColorSyncFlag,
                        device[devID].buildDepthSyncFlag,
                        device[devID].buildConfidenceFlag
                      );

  return 1;
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
 device[devID].pixelsColorAcq = getPixelsColorsAcq();
 device[devID].pixelsDepthSync = getPixelsDepthSync();
 device[devID].pixelsConfidenceQVGA = getPixelsConfidenceQVGA();

 if (device[devID].interpolateDepthFlag)
    {
         device[devID].pixelsDepthAcq = getPixelsDepthAcqVGA();
         device[devID].pixelsColorSync = getPixelsColorSyncVGA();
    } else
    {
         device[devID].pixelsDepthAcq = getPixelsDepthAcqQVGA();
         device[devID].pixelsColorSync = getPixelsColorSyncQVGA();
    }

 device[devID].frameCount = getFrameCount();
 device[devID].timeStamp = getTimeStamp();

  return 1;
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
unsigned long getLastDepthSenseColorTimestamp(int devID) { return device[devID].timeStamp; }
int getDepthSenseColorWidth(int devID)        { return device[devID].widthColor; }
int getDepthSenseColorHeight(int devID)       { return device[devID].heightColor; }
int getDepthSenseColorDataSize(int devID)     { return device[devID].widthColor*device[devID].heightColor * 3; }
int getDepthSenseColorChannels(int devID)     { return 3; }
int getDepthSenseColorBitsPerPixel(int devID) { return 8; }

// Frame Grabber should call this function for color frames
unsigned char * getDepthSenseColorPixels(int devID)    { return device[devID].pixelsColorSync; }


   //Depth Frame getters
unsigned long getLastDepthSenseDepthTimestamp(int devID) { return device[devID].timeStamp; }
int getDepthSenseDepthWidth(int devID)    { return device[devID].widthDepthAcq; }
int getDepthSenseDepthHeight(int devID)   { return device[devID].heightDepthAcq; }
int getDepthSenseDepthDataSize(int devID) { return device[devID].widthDepthAcq*device[devID].heightDepthAcq; }
int getDepthSenseDepthChannels(int devID)     { return 1; }
int getDepthSenseDepthBitsPerPixel(int devID) { return 16; }

// Frame Grabber should call this function for depth frames
char * getDepthSenseDepthPixels(int devID) { return (char *) device[devID].pixelsDepthAcq ; }


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
