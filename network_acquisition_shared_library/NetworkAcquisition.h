#ifndef NETWORKACQUISITION_H_INCLUDED
#define NETWORKACQUISITION_H_INCLUDED

//#include "../acquisition/acquisition_setup.h"
//NetworkAcquisition does not conflict with anything so we default to building it
#define BUILD_NETWORK 1

#define USE_CALIBRATION 1

#if USE_CALIBRATION
#include "../tools/Calibration/calibration.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif


#define MAX_NETWORK_DEVICES 5


struct NetworkVirtualDevice
{
 unsigned int cycle;


 unsigned int colorWidth , colorHeight , colorChannels , colorBitsperpixel;
 unsigned long lastColorTimestamp;
 unsigned long compressedColorSize;
 char * colorFrame;
 volatile int okToSendColorFrame;


 unsigned int depthWidth , depthHeight , depthChannels , depthBitsperpixel;
 unsigned long lastDepthTimestamp;
 short * depthFrame;
 volatile int okToSendDepthFrame;

 struct calibration calibRGB;
 struct calibration calibDepth;
};

extern struct NetworkVirtualDevice networkDevice[MAX_NETWORK_DEVICES];


   int networkBackbone_startPushingToRemote(char * ip , int port);
   int networkBackbone_stopPushingToRemote(int frameServerID);
   int networkBackbone_pushImageToRemote(int frameServerID, int streamNumber , void*  pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);






   //Initialization of Network
   int startNetworkModule(unsigned int max_devs,char * settings);

   #if BUILD_NETWORK
   int getNetworkNumberOfDevices(); // This has to be called AFTER startNetwork
   int stopNetworkModule();

   //Basic Per Device Operations
   int createNetworkDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);
   int destroyNetworkDevice(int devID);

   int seekNetworkFrame(int devID,unsigned int seekFrame);
   int snapNetworkFrames(int devID);


   int getNetworkColorCalibration(int devID,struct calibration * calib);
   int getNetworkDepthCalibration(int devID,struct calibration * calib);

   int setNetworkColorCalibration(int devID,struct calibration * calib);
   int setNetworkDepthCalibration(int devID,struct calibration * calib);

   //Color Frame getters
   unsigned long getLastNetworkColorTimestamp(int devID);

   int getNetworkColorWidth(int devID);
   int getNetworkColorHeight(int devID);
   int getNetworkColorDataSize(int devID);
   int getNetworkColorChannels(int devID);
   int getNetworkColorBitsPerPixel(int devID);
   char * getNetworkColorPixels(int devID);
   double getNetworkColorFocalLength(int devID);
   double getNetworkColorPixelSize(int devID);

   //Depth Frame getters
   unsigned long getLastNetworkDepthTimestamp(int devID);

   int getNetworkDepthWidth(int devID);
   int getNetworkDepthHeight(int devID);
   int getNetworkDepthDataSize(int devID);
   int getNetworkDepthChannels(int devID);
   int getNetworkDepthBitsPerPixel(int devID);

   char * getNetworkDepthPixels(int devID);
   double getNetworkDepthFocalLength(int devID);
   double getNetworkDepthPixelSize(int devID);
   #endif


#ifdef __cplusplus
}
#endif


#endif // NETWORKACQUISITION_H_INCLUDED
