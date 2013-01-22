#ifndef OPENNI1ACQUISITION_H_INCLUDED
#define OPENNI1ACQUISITION_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif
   //Initialization of OpenNI1
   int startOpenNI1(unsigned int max_devs);
   int getOpenNI1NumberOfDevices(); // This has to be called AFTER startOpenNI1
   int stopOpenNI1();

   int mapOpenNI1DepthToRGB(int devID);
   int mapOpenNI1RGBToDepth(int devID);

   //Basic Per Device Operations
   int createOpenNI1Device(int devID,unsigned int width,unsigned int height,unsigned int framerate);
   int destroyOpenNI1Device(int devID);
   int snapOpenNI1Frames(int devID);

   //Color Frame getters
   int getOpenNI1ColorWidth(int devID);
   int getOpenNI1ColorHeight(int devID);
   int getOpenNI1ColorDataSize(int devID);
   int getOpenNI1ColorChannels(int devID);
   int getOpenNI1ColorBitsPerPixel(int devID);
   char * getOpenNI1ColorPixels(int devID);

   double getOpenNI1ColorFocalLength(int devID);
   double getOpenNI1ColorPixelSize(int devID);

   //Depth Frame getters
   int getOpenNI1DepthWidth(int devID);
   int getOpenNI1DepthHeight(int devID);
   int getOpenNI1DepthDataSize(int devID);
   int getOpenNI1DepthChannels(int devID);
   int getOpenNI1DepthBitsPerPixel(int devID);
   char * getOpenNI1DepthPixels(int devID);

   double getOpenNI1DepthFocalLength(int devID);
   double getOpenNI1DepthPixelSize(int devID);
#ifdef __cplusplus
}
#endif

#endif // OPENNI1ACQUISITION_H_INCLUDED
