#ifndef FREENECTACQUISITION_H_INCLUDED
#define FREENECTACQUISITION_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

int startFreenectModule(unsigned int max_devs,char * settings);
int stopFreenectModule();

int mapFreenectDepthToRGB(int devID);

int getFreenectNumberOfDevices();

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

#ifdef __cplusplus
}
#endif

#endif // FREENECTACQUISITION_H_INCLUDED
