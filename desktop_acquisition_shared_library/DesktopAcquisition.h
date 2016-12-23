/** @file DesktopAcqusition.h
 *  @brief The plugin module that provides acquisition from Desktops ( i.e. X11 )
 *
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef DESKTOPCQUISITION_H_INCLUDED
#define DESKTOPCQUISITION_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

   //Initialization of Desktop
   int startDesktopModule(unsigned int max_devs,char * settings);
   //DesktopAcquisition does not conflict with anything so we default to building it

   int getDesktopCapabilities(int devID,int capToAskFor);

   int enableDesktopStream(int devID,unsigned int streamID);
   int disableDesktopStream(int devID,unsigned int streamID);

   int getDesktopNumberOfDevices(); // This has to be called AFTER startDesktop
   int stopDesktopModule();

   int switchDesktopToColorStream(int devID);
   int mapDesktopDepthToRGB(int devID);
   int mapDesktopRGBToDepth(int devID);
   int getDesktopNumberOfColorStreams(int devID);

   //Basic Per Device Operations
   int listDesktopDevices(int devID,char * output, unsigned int maxOutput);
   int createDesktopDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);
   int destroyDesktopDevice(int devID);

   int getTotalDesktopFrameNumber(int devID);
   int getCurrentDesktopFrameNumber(int devID);

   int controlDesktopFlow(int devID,float newFlowState);

   int seekRelativeDesktopFrame(int devID,signed int seekFrame);
   int seekDesktopFrame(int devID,unsigned int seekFrame);
   int snapDesktopFrames(int devID);



   //Color Frame getters
   unsigned long getLastDesktopColorTimestamp(int devID);

   int getDesktopColorWidth(int devID);
   int getDesktopColorHeight(int devID);
   int getDesktopColorDataSize(int devID);
   int getDesktopColorChannels(int devID);
   int getDesktopColorBitsPerPixel(int devID);
   unsigned char * getDesktopColorPixels(int devID);
   double getDesktopColorFocalLength(int devID);
   double getDesktopColorPixelSize(int devID);

   //Depth Frame getters
   unsigned long getLastDesktopDepthTimestamp(int devID);

   int getDesktopDepthWidth(int devID);
   int getDesktopDepthHeight(int devID);
   int getDesktopDepthDataSize(int devID);
   int getDesktopDepthChannels(int devID);
   int getDesktopDepthBitsPerPixel(int devID);

   char * getDesktopDepthPixels(int devID);
   char * getDesktopDepthPixelsFlipped(int devID);

   double getDesktopDepthFocalLength(int devID);
   double getDesktopDepthPixelSize(int devID);


#ifdef __cplusplus
}
#endif

#endif // TEMPLATEACQUISITION_H_INCLUDED
