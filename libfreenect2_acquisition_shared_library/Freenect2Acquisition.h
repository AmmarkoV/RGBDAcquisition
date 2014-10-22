#ifndef FREENECTACQUISITION_H_INCLUDED
#define FREENECTACQUISITION_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

#include "../acquisition/acquisition_setup.h"

int startFreenect2Module(unsigned int max_devs,char * settings);

#if BUILD_FREENECT2
int stopFreenect2Module();

int createFreenect2Device(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);

int mapFreenect2DepthToRGB(int devID);

int getFreenect2NumberOfDevices();

int seekFreenect2Frame(int devID,unsigned int seekFrame);
int snapFreenect2Frames(int devID);

int getFreenect2ColorWidth(int devID);
int getFreenect2ColorHeight(int devID);
int getFreenect2ColorDataSize(int devID);
int getFreenect2ColorChannels(int devID);
int getFreenect2ColorBitsPerPixel(int devID);
char * getFreenect2ColorPixels(int devID);

int getFreenect2DepthWidth(int devID);
int getFreenect2DepthHeight(int devID);
int getFreenect2DepthDataSize(int devID);
int getFreenect2DepthChannels(int devID);
int getFreenect2DepthBitsPerPixel(int devID);
char * getFreenect2DepthPixels(int devID);
#endif

#ifdef __cplusplus
}
#endif

#endif // FREENECTACQUISITION_H_INCLUDED
