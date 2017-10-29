#ifndef REALSENSEACQUISITION_H_INCLUDED
#define REALSENSEACQUISITION_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

#include "../acquisition/acquisition_setup.h"

   #if USE_CALIBRATION
    #include "../tools/Calibration/calibration.h"
   #endif


//#define BUILD_REALSENSE 1
int startRealsenseModule(unsigned int max_devs,const char * settings);

#if BUILD_REALSENSE
int stopRealsenseModule();

int createRealsenseDevice(int devID,const char * devName,unsigned int width,unsigned int height,unsigned int framerate);
int destroyRealsenseDevice(int devID);

int mapRealsenseDepthToRGB(int devID);

int getRealsenseNumberOfDevices();

int seekRealsenseFrame(int devID,unsigned int seekFrame);
int snapRealsenseFrames(int devID);

int getRealsenseColorWidth(int devID);
int getRealsenseColorHeight(int devID);
int getRealsenseColorDataSize(int devID);
int getRealsenseColorChannels(int devID);
int getRealsenseColorBitsPerPixel(int devID);
char * getRealsenseColorPixels(int devID);

int getRealsenseDepthWidth(int devID);
int getRealsenseDepthHeight(int devID);
int getRealsenseDepthDataSize(int devID);
int getRealsenseDepthChannels(int devID);
int getRealsenseDepthBitsPerPixel(int devID);
char * getRealsenseDepthPixels(int devID);

   #if USE_CALIBRATION
    int getRealsenseColorCalibration(int devID,struct calibration * calib);
    int getRealsenseDepthCalibration(int devID,struct calibration * calib);
   #endif

#endif

#ifdef __cplusplus
}
#endif

#endif // FREENECTACQUISITION_H_INCLUDED
