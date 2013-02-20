#ifndef OPENNI2ACQUISITION_H_INCLUDED
#define OPENNI2ACQUISITION_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif
   //Initialization of OpenNI2
   int startOpenNI2(unsigned int max_devs);
   int getOpenNI2NumberOfDevices(); // This has to be called AFTER startOpenNI2
   int stopOpenNI2();

   int mapOpenNI2DepthToRGB(int devID);
   int mapOpenNI2RGBToDepth(int devID);

   //Basic Per Device Operations
   int createOpenNI2Device(int devID,unsigned int width,unsigned int height,unsigned int framerate);
   int destroyOpenNI2Device(int devID);
   int snapOpenNI2Frames(int devID);

   //Color Frame getters
   int getOpenNI2ColorWidth(int devID);
   int getOpenNI2ColorHeight(int devID);
   int getOpenNI2ColorDataSize(int devID);
   int getOpenNI2ColorChannels(int devID);
   int getOpenNI2ColorBitsPerPixel(int devID);
   char * getOpenNI2ColorPixels(int devID);

   double getOpenNI2ColorFocalLength(int devID);
   double getOpenNI2ColorPixelSize(int devID);

   //Depth Frame getters
   int getOpenNI2DepthWidth(int devID);
   int getOpenNI2DepthHeight(int devID);
   int getOpenNI2DepthDataSize(int devID);
   int getOpenNI2DepthChannels(int devID);
   int getOpenNI2DepthBitsPerPixel(int devID);
   short * getOpenNI2DepthPixels(int devID);

   double getOpenNI2DepthFocalLength(int devID);
   double getOpenNI2DepthPixelSize(int devID);

#ifdef __cplusplus
}
#endif

#endif // OPENNI2ACQUISITION_H_INCLUDED
