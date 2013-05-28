#ifndef V4L2ACQUISITION_H_INCLUDED
#define V4L2ACQUISITION_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif
   //Initialization of V4L2
   int startV4L2(unsigned int max_devs,char * settings);
   int getV4L2(); // This has to be called AFTER startV4L2
   int stopV4L2();

   int getDevIDForV4L2Name(char * devName);

   //Basic Per Device Operations
   int createV4L2Device(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);
   int destroyV4L2Device(int devID);

   int seekV4L2Frame(int devID,unsigned int seekFrame);
   int snapV4L2Frames(int devID);

   //Color Frame getters
   int getV4L2ColorWidth(int devID);
   int getV4L2ColorHeight(int devID);
   int getV4L2ColorDataSize(int devID);
   int getV4L2ColorChannels(int devID);
   int getV4L2ColorBitsPerPixel(int devID);
   char * getV4L2ColorPixels(int devID);
   double getV4L2ColorFocalLength(int devID);
   double getV4L2ColorPixelSize(int devID);

   //Depth Frame getters
   int getV4L2DepthWidth(int devID);
   int getV4L2DepthHeight(int devID);
   int getV4L2DepthDataSize(int devID);
   int getV4L2DepthChannels(int devID);
   int getV4L2DepthBitsPerPixel(int devID);

   char * getV4L2DepthPixels(int devID);
   double getV4L2DepthFocalLength(int devID);
   double getV4L2DepthPixelSize(int devID);

#ifdef __cplusplus
}
#endif

#endif // V4L2ACQUISITION_H_INCLUDED
