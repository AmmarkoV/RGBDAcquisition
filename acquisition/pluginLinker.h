/** @file pluginLinker.h
 *  @brief The plugin mechanism which Acquisition uses to reference its modules
 *
 *  This is object orientation for C :D
 *
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug This does not yet correctly scan all possible locations for the plugins requested , LD_LIBRARY_PATH etc..
 */

#ifndef PLUGINLINKER_H_INCLUDED
#define PLUGINLINKER_H_INCLUDED


#include "Acquisition.h"

/*
           This is the mechanism that loads and provides access to the calls of all the plugins that Acquisition uses
*/

/**
 * @brief An enumerator to access the strings that define a plugin , each plugin has a name ( i.e. Template , a Path i.e. ../template_acquisition_shared_library and a library , i.e libTemplateAcquisition.so )
 */
enum pluginStrEnum
{
  PLUGIN_NAME_STR = 0,
  PLUGIN_PATH_STR  ,
  PLUGIN_LIBNAME_STR
};

/**
 * @brief This is the structure that contains the calls for each of the plugins loaded
 *
 * Each of this functions is accessed like plugins[moduleID].FUNCTION  initially all of them are 0
 * When the plugin gets loaded they point to the correct address space
 * Connecting them is done with the int linkToPlugin(char * moduleName,char * modulePossiblePath ,char * moduleLib ,  ModuleIdentifier moduleID) and
 * int unlinkPlugin(ModuleIdentifier moduleID);
 * This works great and greatly simplifies the way
 */
struct acquisitionPluginInterface
{
   void *handle;

   int (*startModule) (unsigned int,char *);
   int (*stopModule) ();

   int (*mapDepthToRGB) (int);
   int (*mapRGBToDepth) (int);


   int (*listDevices) (int,char *,unsigned int);
   int (*changeResolution) (unsigned int,unsigned int);
   int (*createDevice)  (int,char *,unsigned int,unsigned int,unsigned int);
   int (*destroyDevice) (int);


   int (*getNumberOfDevices) ();


   unsigned long (*getLastColorTimestamp) (int);
   unsigned long (*getLastDepthTimestamp) (int);

   int (*snapFrames) (int);


   int (*enableStream)  (int,unsigned int);
   int (*disableStream) (int,unsigned int);

   int (*getTotalFrameNumber)  (int);
   int (*getCurrentFrameNumber)  (int);

   int (*seekRelativeFrame)  (int,signed int);
   int (*seekFrame)  (int,unsigned int);
   int (*controlFlow) (int,float);

   int (*getNumberOfColorStreams) (int);
   int (*switchToColorStream) (int, unsigned int);

   int (*getColorWidth) (int);
   int (*getColorHeight) (int);
   int (*getColorDataSize) (int);
   int (*getColorChannels) (int);
   int (*getColorBitsPerPixel) (int);
   unsigned char * (*getColorPixels) (int);
   //double (*getColorFocalLength) (int);
   //double (*getColorPixelSize)   (int);
   int (*getColorCalibration) (int,struct calibration *);
   int (*setColorCalibration) (int,struct calibration *);


   int (*getDepthWidth) (int);
   int (*getDepthHeight) (int);
   int (*getDepthDataSize) (int);
   int (*getDepthChannels) (int);
   int (*getDepthBitsPerPixel) (int);
   unsigned char * (*getDepthPixels) (int);
   //double (*getDepthFocalLength) (int);
   //double (*getDepthPixelSize)   (int);
   int (*getDepthCalibration) (int,struct calibration *);
   int (*setDepthCalibration) (int,struct calibration *);


   unsigned int forcedWidth;
   unsigned int forcedHeight;
   unsigned int forceResolution;
};

/**
 * @brief The array that holds the functions and the state of each plugin
 *        The elements of the array get populated using int linkToPlugin(char * moduleName,char * modulePossiblePath ,char * moduleLib ,  ModuleIdentifier moduleID);
 *        and emptied using int unlinkPlugin(ModuleIdentifier moduleID);
 */
extern struct acquisitionPluginInterface plugins[NUMBER_OF_POSSIBLE_MODULES];

extern void * remoteNetworkDLhandle;
extern int (*startPushingToRemoteNetwork) (char * , int);
extern int (*stopPushingToRemoteNetwork) (int);
extern int (*pushImageToRemoteNetwork) (int,int,void *,unsigned int,unsigned int,unsigned int,unsigned int);




/**
 * @brief Get one of the plugin strings
 * each plugin has a name ( i.e. Template ) a Path ( i.e. ../template_acquisition_shared_library )
 * and a library that hosts it ( i.e libTemplateAcquisition.so ) , so this function provides access to these strings
 *
 * @ingroup plugin
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param strID , An integer value describing the string to get back ( enum pluginStrEnum )
 * @retval A string , or 0 for failure
 */
char * getPluginStr(int moduleID,int strID);



/**
 * @brief Scan for a library and return the path where it is found
 * @ingroup plugin
 * @param String , a known path to the library
 * @param String , a library name for example ( libTemplateAcquisition.so )
 * @param Pointer , output  to a string to accomodate the path to the library
 * @param Length , of output pointer string
 * @retval 1=Success , 0=Failure
 */
int getPluginPath(char * possiblePath, char * libName , char * pathOut, unsigned int pathOutLength);



/**
 * @brief Checks if a plugin is loaded
 *
 * @ingroup plugin
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval 1=Success , 0=Failure
 */
int isPluginLoaded(ModuleIdentifier moduleID);



/**
 * @brief Link to the network transmission modules
 * @ingroup plugin
 * @param String , module name
 * @param String , a possible path to the library
 * @param String , the name of the module library
 * @retval 1=Success , 0=Failure
 */
int linkToNetworkTransmission(char * moduleName,char * modulePossiblePath ,char * moduleLib);


/**
 * @brief Link to a plugin module
 * This will make plugins[moduleID].startModule() point to startTemplateModule() for example
 * @ingroup plugin
 * @param String , module name
 * @param String , a possible path to the library
 * @param String , the name of the module library
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval 1=Success , 0=Failure
 */
int linkToPlugin(char * moduleName,char * modulePossiblePath ,char * moduleLib ,  ModuleIdentifier moduleID);


/**
 * @brief Unlink a plugin module
 * This will make plugins[moduleID].startModule point to zero for example
 * @ingroup plugin
 * @param String , module name
 * @param String , a possible path to the library
 * @param String , the name of the module library
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval 1=Success , 0=Failure
 */
int unlinkPlugin(ModuleIdentifier moduleID);

#endif // PLUGINLINKER_H_INCLUDED
