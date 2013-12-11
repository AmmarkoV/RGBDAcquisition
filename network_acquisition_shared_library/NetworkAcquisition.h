/** @file NetworkAcqusition.h
 *  @brief The plugin module that provides acquisition from Network streams and broadcasts to network
 *
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug This library does not yet work , it is work under progress
 */


#ifndef NETWORKACQUISITION_H_INCLUDED
#define NETWORKACQUISITION_H_INCLUDED

#define USE_CALIBRATION 1

#if USE_CALIBRATION
#include "../tools/Calibration/calibration.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

#include "../acquisition/acquisition_setup.h"

#define MAX_NETWORK_DEVICES 5


struct NetworkVirtualDevice
{
 unsigned int cycle;


 unsigned int colorWidth , colorHeight , colorChannels , colorBitsperpixel;
 unsigned long lastColorTimestamp;
 unsigned long compressedColorSize;
 unsigned char * colorFrame;
 volatile int okToSendColorFrame;


 unsigned int depthWidth , depthHeight , depthChannels , depthBitsperpixel;
 unsigned long lastDepthTimestamp;
 unsigned short * depthFrame;
 volatile int okToSendDepthFrame;

 struct calibration calibRGB;
 struct calibration calibDepth;
};

extern struct NetworkVirtualDevice networkDevice[MAX_NETWORK_DEVICES];


   int networkBackbone_startPushingToRemote(char * ip , int port);
   int networkBackbone_stopPushingToRemote(int frameServerID);
   int networkBackbone_pushImageToRemote(int frameServerID, int streamNumber , void*  pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);






   //Initialization of Network

  /**
   * @brief Print a countdown in console and then return
   * @ingroup networkAcquisition
   * @retval 1=Success , 0=Failure
   * @bug Network Acquisition Module is work under progress ,it is currently non-operational
   */
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
   unsigned char * getNetworkColorPixels(int devID);
   double getNetworkColorFocalLength(int devID);
   double getNetworkColorPixelSize(int devID);

   //Depth Frame getters
   unsigned long getLastNetworkDepthTimestamp(int devID);

   int getNetworkDepthWidth(int devID);
   int getNetworkDepthHeight(int devID);
   int getNetworkDepthDataSize(int devID);
   int getNetworkDepthChannels(int devID);
   int getNetworkDepthBitsPerPixel(int devID);

   unsigned char * getNetworkDepthPixels(int devID);
   double getNetworkDepthFocalLength(int devID);
   double getNetworkDepthPixelSize(int devID);
   #endif


#ifdef __cplusplus
}
#endif


#endif // NETWORKACQUISITION_H_INCLUDED
