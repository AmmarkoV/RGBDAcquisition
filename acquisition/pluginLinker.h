#ifndef PLUGINLINKER_H_INCLUDED
#define PLUGINLINKER_H_INCLUDED


#include "Acquisition.h"

/*
           This is the mechanism that loads and provides access to the calls of all the plugins that Acquisition uses
*/

enum pluginStrEnum
{
  PLUGIN_NAME_STR = 0,
  PLUGIN_PATH_STR  ,
  PLUGIN_LIBNAME_STR
};


struct acquisitionPluginInterface
{
   void *handle;

   int (*startModule) (unsigned int,char *);
   int (*stopModule) ();

   int (*mapDepthToRGB) (int);
   int (*mapRGBToDepth) (int);


   int (*listDevices) (int,char *,unsigned int);
   int (*createDevice)  (int,char *,unsigned int,unsigned int,unsigned int);
   int (*destroyDevice) (int);


   int (*getNumberOfDevices) ();


   unsigned long (*getLastColorTimestamp) (int);
   unsigned long (*getLastDepthTimestamp) (int);

   int (*snapFrames) (int);


   int (*getTotalFrameNumber)  (int);
   int (*getCurrentFrameNumber)  (int);

   int (*seekRelativeFrame)  (int,signed int);
   int (*seekFrame)  (int,unsigned int);

   int (*getColorWidth) (int);
   int (*getColorHeight) (int);
   int (*getColorDataSize) (int);
   int (*getColorChannels) (int);
   int (*getColorBitsPerPixel) (int);
   char * (*getColorPixels) (int);
   double (*getColorFocalLength) (int);
   double (*getColorPixelSize)   (int);
   int (*getColorCalibration) (int,struct calibration *);
   int (*setColorCalibration) (int,struct calibration *);


   int (*getDepthWidth) (int);
   int (*getDepthHeight) (int);
   int (*getDepthDataSize) (int);
   int (*getDepthChannels) (int);
   int (*getDepthBitsPerPixel) (int);
   char * (*getDepthPixels) (int);
   double (*getDepthFocalLength) (int);
   double (*getDepthPixelSize)   (int);
   int (*getDepthCalibration) (int,struct calibration *);
   int (*setDepthCalibration) (int,struct calibration *);


};

extern struct acquisitionPluginInterface plugins[NUMBER_OF_POSSIBLE_MODULES];

extern void * remoteNetworkDLhandle;
extern int (*startPushingToRemoteNetwork) (char * , int);
extern int (*stopPushingToRemoteNetwork) (int);
extern int (*pushImageToRemoteNetwork) (int,int,void *,unsigned int,unsigned int,unsigned int,unsigned int);


char * getPluginStr(int moduleID,int strID);
int getPluginPath(char * possiblePath, char * libName , char * pathOut, unsigned int pathOutLength);

int linkToNetworkTransmission(char * moduleName,char * modulePossiblePath ,char * moduleLib);

int isPluginLoaded(ModuleIdentifier moduleID);

int linkToPlugin(char * moduleName,char * modulePossiblePath ,char * moduleLib ,  ModuleIdentifier moduleID);

int unlinkPlugin(ModuleIdentifier moduleID);

#endif // PLUGINLINKER_H_INCLUDED
