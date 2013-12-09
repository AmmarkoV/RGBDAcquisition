/** @file Acqusition.h
 *  @brief The main Acquisition library that handles plugins and provides .
 *
 * Any application that may want to interface with RGBDAcquition will probably want to link to libAcquisition.so
 * and include this header. It provides the entry point for acquisition and internally loads/unloads all the existing
 * sub-modules on runtime.
 * Basic usage is the following
 * acquisitionStartModule(ModuleIdentifier moduleID,unsigned int maxDevices,char * settings); to initialize start a module of up to maxDevices devices
 * acquisitionOpenDevice(ModuleIdentifier moduleID,DeviceIdentifier devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);
 * This is pretty much self documenting , so you just open a specific device of the module you have already initialized
 *
 * while (1)
 * {
 *
 * unsigned char * acquisitionGetColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID);
 * unsigned short * acquisitionGetDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID);
 *
 * }
 *
 * acquisitionCloseDevice(ModuleIdentifier moduleID,DeviceIdentifier devID);
 * acquisitionStopModule(ModuleIdentifier moduleID);
 *
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug This is not yet thread safe
 */


#ifndef ACQUISITION_H_INCLUDED
#define ACQUISITION_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif


#include "../tools/Calibration/calibration.h"



/**
 * @brief An enumerator for the possible modules to be used , possible values for moduleID
 *
 * This enumerator holds all the possible values passed when and where moduleID is requested
 * These have an 1:1 relation to the modules that are supported and are used to reference them
 *
 */
enum Acquisition_Possible_Modules
{
    NO_ACQUISITION_MODULE = 0,
    V4L2_ACQUISITION_MODULE ,
    V4L2STEREO_ACQUISITION_MODULE ,
    FREENECT_ACQUISITION_MODULE ,
    OPENNI1_ACQUISITION_MODULE  ,
    OPENNI2_ACQUISITION_MODULE  ,
    OPENGL_ACQUISITION_MODULE  ,
    TEMPLATE_ACQUISITION_MODULE  ,
    NETWORK_ACQUISITION_MODULE  ,
   //--------------------------------------------
    NUMBER_OF_POSSIBLE_MODULES
};

#define NUMBER_OF_POSSIBLE_DEVICES 20


typedef unsigned int ModuleIdentifier;
typedef unsigned int DeviceIdentifier;


/**
 * @brief A structure that holds the state of a specific device of a module
 *
 * A structure to hold the state of each of the devices initialized by the clients
 *
 */
struct acquisitionDeviceStates
{
  char outputString[1024];

  //File output
  unsigned char fileOutput;

  //Network output
  unsigned char networkOutput;
  int frameServerID ;
  int port;

  //Dry Run
  unsigned char dryRunOutput;
};


/**
 * @brief A structure to hold the state of each of the devices initialized by the clients
 *
 * This structure holds an array of all devices for each module
 *
 */
struct acquisitionModuleStates
{
  struct acquisitionDeviceStates device[NUMBER_OF_POSSIBLE_DEVICES];
};



/**
 * @brief Find if file pointed by filename path exists
 * @ingroup misc
 * @param filename string
 * @retval 1 if file Exists , 0 if file does not exist
 */
int acquisitionFileExists(char * filename);


/**
 * @brief Start one of the timers in our timer pool
 * @ingroup misc
 * @param A number that specifies a timer , should be 0-10
 * @retval This function does not have a return value
 */
void acquisitionStartTimer(unsigned int timerID);


/**
 * @brief Stop one of the timers in our timer pool
 * @ingroup misc
 * @param A number that specifies a timer , should be 0-10
 * @retval The number of milliseconds elapsed since we started our timer
 */
unsigned int acquisitionStopTimer(unsigned int timerID);


/**
 * @brief Get Frames per Second for a Stopped timer
 * @ingroup misc
 * @param A number that specifies a timer , should be 0-10
 * @retval A float of the frames per seconds since we started our timer
 */
float acquisitionGetTimerFPS(unsigned int timerID);

/**
 * @brief Print a countdown in console and then return
 * @ingroup misc
 * @param A number of seconds to countdown
 * @retval This function does not have a return value
 */
void countdownDelay(int seconds);

/**
 * @brief Acquisition library has an internal timer that gets passed to frame timestamps. This function can force this internal timer to a specific value , all Target Frames until the next call of this function will have the value passed here
 * @ingroup acquisitionCore
 * @param Number Of Milliseconds to set
 * @retval 1 = Success , 0 = Failure
 */
int acquisitionSimulateTime(unsigned long timeInMillisecs);

/**
 * @brief  Save a Frame Buffer to a file
 * @ingroup acquisitionCore
 * @param String containing the target Filename
 * @param Pointer to the buffer that holds the image we want to store
 * @param Width in pixels
 * @param Height in pixels
 * @param The number of color channels of the buffer , 3 for RGB , 1 for Depth
 * @param The number of bits per pixel 24 for RGB ( 8 bytes for 3 channels ) , 16 for Depth ( 16 bytes for 1 channel )
 * @retval 1 = Success saving the file  , 0 = Failure
 */
int saveRawImageToFile(char * filename,unsigned char * pixels , unsigned int width , unsigned int height , unsigned int channels , unsigned int bitsperpixel);

/**
 * @brief  Convert a Depth Frame Buffer to 3channel 24bit format
 * @ingroup acquisitionCore
 * @param Unsigned Short 16bit 1channel depths
 * @param Width in pixels
 * @param Height in pixels
 * @retval 0 = Failure , On Success a pointer to buffer containing the converted Depth is returned , dont forget to free it when done
 */
unsigned char * convertShortDepthToRGBDepth(unsigned short * depth,unsigned int width , unsigned int height);

/**
 * @brief  Convert a Depth Frame Buffer to 3channel 24bit format
 * @ingroup acquisitionCore
 * @param Unsigned Short 16bit 1channel depths
 * @param Width in pixels
 * @param Height in pixels
 * @param Minimum Depth Value
 * @param Maximum Depth Value
 * @retval 0 = Failure , On Success a pointer to buffer containing the converted Depth is returned , dont forget to free it when done
 */
unsigned char * convertShortDepthToCharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);

/**
 * @brief  Convert a Depth Frame Buffer to 3channel 24bit format
 * @ingroup acquisitionCore
 * @param Unsigned Short 16bit 1channel depths
 * @param Width in pixels
 * @param Height in pixels
 * @param Minimum Depth Value
 * @param Maximum Depth Value
 * @retval 0 = Failure , On Success a pointer to buffer containing the converted Depth is returned , dont forget to free it when done
 */
unsigned char * convertShortDepthTo3CharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);

/**
 * @brief  Converts a module name like "TEMPLATE" to its proper moduleID TEMPLATE_ACQUISITION_MODULE
   See enum Acquisition_Possible_Modules for the possible results and the body of the function for conventions used for the string input
 * @ingroup acquisitionCore
 * @param String with the friendly name of a module
 * @retval 0 or NO_ACQUISITION_MODULE means that input was incorrect , On Success on of enum Acquisition_Possible_Modules values will be returned
 */
ModuleIdentifier getModuleIdFromModuleName(char * moduleName);


/**
 * @brief  Converts a  moduleID like TEMPLATE_ACQUISITION_MODULE to its module name like "TEMPLATE"
 * @ingroup acquisitionCore
 * @param  moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval String with the friendly name of a module
 */
char * getModuleNameFromModuleID(ModuleIdentifier moduleID);


/**
 * @brief  Return the number of Modules that can be linked by the running instance of this build of RGBDAcquisition
 * @ingroup acquisitionCore
 * @retval The number of Modules that can be linked by the running instance of this build of RGBDAcquisition
 */
int acquisitionGetModulesCount();

/**
 * @brief  Check to see if this moduleID is availiable , and it is possible to be linked and loaded
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval 1=Yes , 0=No ,No means you wont be able to use this module at all!
 */
int acquisitionIsModuleAvailiable(ModuleIdentifier moduleID);


/**
 * @brief  Check to see if this moduleID has been loaded
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval 1=Yes , 0=No ,No means you wont be able to use this module at all!
 */
int acquisitionPluginIsLoaded(ModuleIdentifier moduleID);


/**
 * @brief  Load the plugin to enable a module specified by moduleID
 * Please note that this function gets automatically called and is only exposed for when someone wants to do more advanced things with the plugin mechanism
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval 1=Success , 0=Failure ,No means you wont be able to use this module at all!
 */
int acquisitionLoadPlugin(ModuleIdentifier moduleID);


/**
 * @brief  Unload the plugin to stop using a module specified by moduleID
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval 1=Success , 0=Failure ,No means we failed to unload
 */
int acquisitionUnloadPlugin(ModuleIdentifier moduleID);



/**
 * @brief  Check to see if this moduleID/devID combination can be a virtual device..
 * @ingroup acquisitionCore
 * @retval 1=Yes , 0=No
 * @bug This is not thoroughly implemented , its just a heuristic check
 */
int acquisitionMayBeVirtualDevice(ModuleIdentifier moduleID,DeviceIdentifier devID , char * devName);


/**
 * @brief  Initialize a module , the module will be automatically loaded and get ready for a subsequent acquisitionOpenDevice call
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param Maximum Devices to be used
 * @param A string that passes module specific settings , can also be null
 * @retval 1=Success , 0=Failure
 */
int acquisitionStartModule(ModuleIdentifier moduleID,unsigned int maxDevices,char * settings);

/**
 * @brief  Stop using a module , the module will be automatically unloaded , no further calls should be made to this module ( and if they are made they will fail )
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval 1=Success , 0=Failure
 */
int acquisitionStopModule(ModuleIdentifier moduleID);

/**
 * @brief  Get the number of devices that can be accessed for this moduleID
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @retval  The number of availiable devices or 0 for Failure or No Devices availiable
 */
int acquisitionGetModuleDevices(ModuleIdentifier moduleID);

/**
 * @brief  Fill the output buffer with up to maxOuput characters of comma seperated strings that describe possible devNames for this module
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param A pointer to a char * buffer that will hold the output
 * @param The size of the output buffer
 * @retval  1 = Success , 0=Failure
 */
int acquisitionListDevices(ModuleIdentifier moduleID,DeviceIdentifier devID,char * output, unsigned int maxOutput);

/**
 * @brief  Open a device with specified dimensions and framerate ,  devNames might be used to identify a specific device by its friendly name.
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param This is a string that might select a device using its friendly name , can also be null
 * @param Width in pixels that we *WISH* to have ( this might not be what the device returns , please call acquisitionGetColorFrameDimensions / acquisitionGetDepthFrameDimensions to find out )
 * @param Height in pixels that we *WISH* to have ( this might not be what the device returns , please call acquisitionGetColorFrameDimensions / acquisitionGetDepthFrameDimensions to find out )
 * @param The framerate we *WISH* to have ( this might not be what the device returns )
 * @retval  1 = Success , 0=Failure
 */
int acquisitionOpenDevice(ModuleIdentifier moduleID,DeviceIdentifier devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate);

/**
 * @brief  Gracefully close an already opened device
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @retval    1 = Success , 0=Failure
 */
int acquisitionCloseDevice(ModuleIdentifier moduleID,DeviceIdentifier devID);

/**
 * @brief  Get the total number of frames on the device , if this is zero it means we have a live stream of unknown total frames ( or no stream at all )
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @retval  0=Live Stream or no Stream Availiable , Any other number is a valid total number of frames availiable
 */
int acquisitionGetTotalFrameNumber(ModuleIdentifier moduleID,DeviceIdentifier devID);

/**
 * @brief  Get the number of the current frames on the device , if this is zero it means we have a live stream of unknown total frames ( or no stream at all )
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @retval  0=Live Stream or no Stream Availiable , Any other number is a valid number of the current frames
 */
int acquisitionGetCurrentFrameNumber(ModuleIdentifier moduleID,DeviceIdentifier devID);

/**
 * @brief  Seek a specific frame from the device,  if the device has a live stream this won't do anything
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param The number of the frame we want to seek to
 * @retval  1=Success 0=Failure
 */
int acquisitionSeekFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int seekFrame);

/**
 * @brief  Seek a specific frame from the device relative to the current frame,  if the device has a live stream this won't do anything
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param The number of the frame we want to seek to ( signed integer )
 * @retval  1=Success 0=Failure
 */
int acquisitionSeekRelativeFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,signed int seekFrame);

/**
 * @brief  Snap a new frame from device , this will populate  acquisitionGetColorFrame acquisitionGetDepthFrame with new frames
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @retval  1=Success 0=Failure
 */
int acquisitionSnapFrames(ModuleIdentifier moduleID,DeviceIdentifier devID);

/**
 * @brief  Get color calibration for this device ( see struct calibration )
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param Calibration to be populated with the information of the Device
 * @retval  1=Success 0=Failure
 */
int acquisitionGetColorCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib);
/**
 * @brief  Get depth calibration for this device ( see struct calibration )
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param Calibration to be populated with the information of the Device
 * @retval  1=Success 0=Failure
 */
int acquisitionGetDepthCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib);

/**
 * @brief  Set color calibration for this device ( see struct calibration )
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param Calibration to be used for this Device
 * @retval  1=Success 0=Failure
 */
int acquisitionSetColorCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib);
/**
 * @brief  Set depth calibration for this device ( see struct calibration )
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param Calibration to be used for this Device
 * @retval  1=Success 0=Failure
 */
int acquisitionSetDepthCalibration(ModuleIdentifier moduleID,DeviceIdentifier devID,struct calibration * calib);

/**
 * @brief  Returns the timestamp value of the last snapped Color Frame , time should be in milliseconds
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @retval  Milliseconds timestamp
 */
unsigned long acquisitionGetColorTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID);

/**
 * @brief  Returns the timestamp value of the last snapped Depth Frame , time should be in milliseconds
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @retval  Milliseconds timestamp
 */
unsigned long acquisitionGetDepthTimestamp(ModuleIdentifier moduleID,DeviceIdentifier devID);

/**
 * @brief  TODO: Overrides current color frame so the next time acquisitionGetColor is called or someone does acquisitionPassFramesToTarget it will be the frame provided there
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param Pointer to the new frame that should overwrite returned , the user should probably free the frame after this call
 * @retval  1=Success 0=Failure
 */
int acquisitionOverrideColorFrame(ModuleIdentifier moduleID , DeviceIdentifier devID , unsigned char * newColor);

/**
 * @brief  Returns the color frame from the last Snap we did using acquisitionSnapFrames to find out its dimensions you might want to call acquisitionGetColorFrameDimensions
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @retval  Pointer to the current color frame ( you shouldnt manually free it) , 0 = Failure
 */
unsigned char * acquisitionGetColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID);
unsigned int acquisitionCopyColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned char * mem,unsigned int memlength);
unsigned int acquisitionCopyColorFramePPM(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned char * mem,unsigned int memlength);

/**
 * @brief  TODO: Overrides current depth frame so the next time acquisitionGetDepth is called or someone does acquisitionPassFramesToTarget it will be the frame provided there
 * @ingroup acquisitionCore
 * @param moduleID , An integer value describing a module ( see enum Acquisition_Possible_Modules )
 * @param deviceID , An integer value that describes one of the possible devices to be used for the module specified by moduleID
 * @param Pointer to the new frame that should overwrite returned , the user should probably free the frame after this call
 * @retval  1=Success 0=Failure
 */
int acquisitionOverrideDepthFrame(ModuleIdentifier moduleID , DeviceIdentifier devID , unsigned short * newDepth);
unsigned short * acquisitionGetDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID);
unsigned int acquisitionCopyDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned short * mem,unsigned int memlength);
unsigned int acquisitionCopyDepthFramePPM(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned short * mem,unsigned int memlength);

int acquisitionGetDepth3DPointAtXYCameraSpace(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int x2d, unsigned int y2d , float *x, float *y , float *z  );
int acquisitionGetDepth3DPointAtXY(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int x2d, unsigned int y2d , float *x, float *y , float *z  );
int acquisitionGetColorFrameDimensions(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int * width , unsigned int * height , unsigned int * channels , unsigned int * bitsperpixel );
int acquisitionGetDepthFrameDimensions(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int * width , unsigned int * height , unsigned int * channels , unsigned int * bitsperpixel );



int acquisitionSavePCDPointCoud(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);
int acquisitionSaveColorFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);
int acquisitionSaveDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);
int acquisitionSaveDepthFrame1C(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);
int acquisitionSaveColoredDepthFrame(ModuleIdentifier moduleID,DeviceIdentifier devID,char * filename);

/* Deprecated
double acqusitionGetColorFocalLength(ModuleIdentifier moduleID,DeviceIdentifier devID);
double acqusitionGetColorPixelSize(ModuleIdentifier moduleID,DeviceIdentifier devID);

double acqusitionGetDepthFocalLength(ModuleIdentifier moduleID,DeviceIdentifier devID);
double acqusitionGetDepthPixelSize(ModuleIdentifier moduleID,DeviceIdentifier devID);
*/

int acquisitionMapDepthToRGB(ModuleIdentifier moduleID,DeviceIdentifier devID);
int acquisitionMapRGBToDepth(ModuleIdentifier moduleID,DeviceIdentifier devID);



/*   ------------------------------------------------------------
        Acquisition transmission to other machines / and files
     ------------------------------------------------------------ */
/**
 * @brief Initialize Target for writing a stream to it, It can be file , network , or loopback output
 * @ingroup target
 * @param moduleID ( see enum Acquisition_Possible_Modules )
 * @param devID the number of the device we want to use
 * @param Target is a string that can be tcp://ip:port , /dev/null/ , or a filename . Depending on its value it initializes network , loopback , or file output
 * @retval 1=Success , 0=Failure
 */
int acquisitionInitiateTargetForFrames(ModuleIdentifier moduleID,DeviceIdentifier devID,char * target);

/**
 * @brief Gracefully stop writing to Target of stream
 * @ingroup target
 * @param moduleID ( see enum Acquisition_Possible_Modules )
 * @param devID the number of the device we want to use
 * @retval 1=Success , 0=Failure
 */
int acquisitionStopTargetForFrames(ModuleIdentifier moduleID,DeviceIdentifier devID);

/**
 * @brief Save Color/Depth streams to the Initialized Target , It can be file , network , or loopback output
 * @ingroup target
 * @param moduleID ( see enum Acquisition_Possible_Modules )
 * @param devID the number of the device we want to use
 * @param Incremental counter that specifies the number of this frame
 * @retval 1=Success , 0=Failure
 */
int acquisitionPassFramesToTarget(ModuleIdentifier moduleID,DeviceIdentifier devID,unsigned int frameNumber);




#ifdef __cplusplus
}
#endif

#endif // ACQUISITION_H_INCLUDED
