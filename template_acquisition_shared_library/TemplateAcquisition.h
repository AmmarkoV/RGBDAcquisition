#ifndef TEMPLATEACQUISITION_H_INCLUDED
#define TEMPLATEACQUISITION_H_INCLUDED



#define USE_CALIBRATION 1

#if USE_CALIBRATION
#include "../tools/Calibration/calibration.h"
#endif

#ifdef __cplusplus
extern "C"
{
#endif

   //Initialization of Template
   int startTemplateModule(unsigned int max_devs,char * settings);
   //TemplateAcquisition does not conflict with anything so we default to building it
   #define BUILD_TEMPLATE 1

   #if BUILD_TEMPLATE
   int getTemplateNumberOfDevices(); // This has to be called AFTER startTemplate
   int stopTemplateModule();

   //Basic Per Device Operations
   int createTemplateDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);
   int destroyTemplateDevice(int devID);

   int getTotalTemplateFrameNumber(int devID);
   int getCurrentTemplateFrameNumber(int devID);


   int seekRelativeTemplateFrame(int devID,signed int seekFrame);
   int seekTemplateFrame(int devID,unsigned int seekFrame);
   int snapTemplateFrames(int devID);


   int getTemplateColorCalibration(int devID,struct calibration * calib);
   int getTemplateDepthCalibration(int devID,struct calibration * calib);

   int setTemplateColorCalibration(int devID,struct calibration * calib);
   int setTemplateDepthCalibration(int devID,struct calibration * calib);

   //Color Frame getters
   unsigned long getLastTemplateColorTimestamp(int devID);

   int getTemplateColorWidth(int devID);
   int getTemplateColorHeight(int devID);
   int getTemplateColorDataSize(int devID);
   int getTemplateColorChannels(int devID);
   int getTemplateColorBitsPerPixel(int devID);
   char * getTemplateColorPixels(int devID);
   double getTemplateColorFocalLength(int devID);
   double getTemplateColorPixelSize(int devID);

   //Depth Frame getters
   unsigned long getLastTemplateDepthTimestamp(int devID);

   int getTemplateDepthWidth(int devID);
   int getTemplateDepthHeight(int devID);
   int getTemplateDepthDataSize(int devID);
   int getTemplateDepthChannels(int devID);
   int getTemplateDepthBitsPerPixel(int devID);

   char * getTemplateDepthPixels(int devID);
   char * getTemplateDepthPixelsFlipped(int devID);

   double getTemplateDepthFocalLength(int devID);
   double getTemplateDepthPixelSize(int devID);
   #endif


#ifdef __cplusplus
}
#endif

#endif // TEMPLATEACQUISITION_H_INCLUDED
