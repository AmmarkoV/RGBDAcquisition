#ifndef V4L2STEREOACQUISITION_H_INCLUDED
#define V4L2STEREOACQUISITION_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif

   #include "../acquisition/acquisition_setup.h"

   //Initialization of V4L2
   int startV4L2StereoModule(unsigned int max_devs,char * settings);

   #if BUILD_V4L2

   int getV4L2StereoCapabilities(int devID,int capToAskFor);

   int getV4L2Stereo(); // This has to be called AFTER startV4L2Stereo
   int stopV4L2StereoModule();

   int getV4L2StereoNumberOfDevices();

   int getDevIDForV4L2StereoName(char * devName);

   //Basic Per Device Operations
   int createV4L2StereoDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);
   int destroyV4L2StereoDevice(int devID);

   int seekV4L2StereoFrame(int devID,unsigned int seekFrame);
   int snapV4L2StereoFrames(int devID);

   int getV4L2StereoNumberOfColorStreams(int devID);
   int switchV4L2StereoToColorStream(int devID,unsigned int streamToActivate);

   //Color Frame getters
   int getV4L2StereoColorWidth(int devID);
   int getV4L2StereoColorHeight(int devID);
   int getV4L2StereoColorDataSize(int devID);
   int getV4L2StereoColorChannels(int devID);
   int getV4L2StereoColorBitsPerPixel(int devID);
   unsigned char * getV4L2StereoColorPixels(int devID);
   unsigned char * getV4L2StereoColorPixelsLeft(int devID);
   unsigned char * getV4L2StereoColorPixelsRight(int devID);

   double getV4L2StereoColorFocalLength(int devID);
   double getV4L2StereoColorPixelSize(int devID);

   //Depth Frame getters
   int getV4L2StereoDepthWidth(int devID);
   int getV4L2StereoDepthHeight(int devID);
   int getV4L2StereoDepthDataSize(int devID);
   int getV4L2StereoDepthChannels(int devID);
   int getV4L2StereoDepthBitsPerPixel(int devID);

   char * getV4L2StereoDepthPixels(int devID);
   double getV4L2StereoDepthFocalLength(int devID);
   double getV4L2StereoDepthPixelSize(int devID);
   #endif

#ifdef __cplusplus
}
#endif

#endif // V4L2STEREOACQUISITION_H_INCLUDED
