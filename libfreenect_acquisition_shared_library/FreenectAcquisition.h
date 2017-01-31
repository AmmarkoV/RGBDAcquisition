#ifndef FREENECTACQUISITION_H_INCLUDED
#define FREENECTACQUISITION_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

#include "../acquisition/acquisition_setup.h"

#if USE_CALIBRATION
    #include "../tools/Calibration/calibration.h"
#endif


int startFreenectModule(unsigned int max_devs,char * settings);

#if BUILD_FREENECT

double getFreenectColorPixelSize(int devID);
double getFreenectColorFocalLength(int devID);
double getFreenectDepthPixelSize(int devID);
double getFreenectDepthFocalLength(int devID);

int stopFreenectModule();

int createFreenectDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);

int mapFreenectDepthToRGB(int devID);

int getFreenectNumberOfDevices();

int seekFreenectFrame(int devID,unsigned int seekFrame);
int snapFreenectFrames(int devID);

int getFreenectColorWidth(int devID);
int getFreenectColorHeight(int devID);
int getFreenectColorDataSize(int devID);
int getFreenectColorChannels(int devID);
int getFreenectColorBitsPerPixel(int devID);
char * getFreenectColorPixels(int devID);

int getFreenectDepthWidth(int devID);
int getFreenectDepthHeight(int devID);
int getFreenectDepthDataSize(int devID);
int getFreenectDepthChannels(int devID);
int getFreenectDepthBitsPerPixel(int devID);
char * getFreenectDepthPixels(int devID);


int getTotalFreenectFrameNumber(int devID);
int getCurrentFreenectFrameNumber(int devID);


#if USE_CALIBRATION
 int getFreenectColorCalibration(int devID,struct calibration * calib);
 int getFreenectDepthCalibration(int devID,struct calibration * calib);
 int setFreenectColorCalibration(int devID,struct calibration * calib);
 int setFreenectDepthCalibration(int devID,struct calibration * calib);
#endif

#endif

#ifdef __cplusplus
}
#endif

#endif // FREENECTACQUISITION_H_INCLUDED
