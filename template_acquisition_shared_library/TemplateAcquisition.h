#ifndef TEMPLATEACQUISITION_H_INCLUDED
#define TEMPLATEACQUISITION_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif
   //Initialization of Template
   int startTemplate(unsigned int max_devs);
   int getTemplateNumberOfDevices(); // This has to be called AFTER startTemplate
   int stopTemplate();

   //Basic Per Device Operations
   int createTemplateDevice(int devID,unsigned int width,unsigned int height,unsigned int framerate);
   int destroyTemplateDevice(int devID);
   int snapTemplateFrames(int devID);

   //Color Frame getters
   int getTemplateColorWidth(int devID);
   int getTemplateColorHeight(int devID);
   int getTemplateColorDataSize(int devID);
   int getTemplateColorChannels(int devID);
   int getTemplateColorBitsPerPixel(int devID);
   char * getTemplateColorPixels(int devID);
   double getTemplateColorFocalLength(int devID);
   double getTemplateColorPixelSize(int devID);

   //Depth Frame getters
   int getTemplateDepthWidth(int devID);
   int getTemplateDepthHeight(int devID);
   int getTemplateDepthDataSize(int devID);
   int getTemplateDepthChannels(int devID);
   int getTemplateDepthBitsPerPixel(int devID);

   char * getTemplateDepthPixels(int devID);
   double getTemplateDepthFocalLength(int devID);
   double getTemplateDepthPixelSize(int devID);

#ifdef __cplusplus
}
#endif

#endif // TEMPLATEACQUISITION_H_INCLUDED
