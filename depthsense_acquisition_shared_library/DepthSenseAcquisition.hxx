/** @file DepthSenseAcqusition.h
 *  @brief The plugin module that provides acquisition from DepthSense Sensors ( i.e. Prerecorded datasets )
 *         In order to make these sensors run you will need to
           0) get a DepthSense camera ( a DS325 is tested with this code )
           1) get and install DepthSenseSDK-1.4.5.2151-amd64-gcc-tar.run or another similar version from the company site http://www.softkinetic.com/Support/Download?EntryId=516
           2) get ph4m's DepthSense grabber , see 3dparty/get_third_party_libs.sh , you basically need to git clone https://github.com/ph4m/DepthSenseGrabber inside the 3dparty directory
           3) ENABLE_DEPTHSENSE through cmake to build the acquisition module
 *
 *  @author Ammar Qammaz (AmmarkoV)
 *  @bug There needs to be substantial cleaning of the DepthSenseAcquisition module
 */

#ifndef DEPTHSENSEACQUISITION_H_INCLUDED
#define DEPTHSENSEACQUISITION_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

   #include "../acquisition/acquisition_setup.h"

   #if USE_CALIBRATION
    #include "../tools/Calibration/calibration.h"
   #endif

   //Initialization of DepthSense
   int startDepthSenseModule(unsigned int max_devs,const char * settings);
   //DepthSenseAcquisition does not conflict with anything so we default to building it
   #define BUILD_DEPTHSENSE 1

   #if BUILD_DEPTHSENSE
   int getDepthSenseCapabilities(int devID,int capToAskFor);

   int enableDepthSenseStream(int devID,unsigned int streamID);
   int disableDepthSenseStream(int devID,unsigned int streamID);

   int getDepthSenseNumberOfDevices(); // This has to be called AFTER startDepthSense
   int stopDepthSenseModule();

   int switchDepthSenseToColorStream(int devID);
   int mapDepthSenseDepthToRGB(int devID);
   int mapDepthSenseRGBToDepth(int devID);
   int getDepthSenseNumberOfColorStreams(int devID);

   //Basic Per Device Operations
   int listDepthSenseDevices(int devID,char * output, unsigned int maxOutput);
   int createDepthSenseDevice(int devID,const char * devName,unsigned int width,unsigned int height,unsigned int framerate);
   int destroyDepthSenseDevice(int devID);

   int getTotalDepthSenseFrameNumber(int devID);
   int getCurrentDepthSenseFrameNumber(int devID);

   int controlDepthSenseFlow(int devID,float newFlowState);

   int seekRelativeDepthSenseFrame(int devID,signed int seekFrame);
   int seekDepthSenseFrame(int devID,unsigned int seekFrame);
   int snapDepthSenseFrames(int devID);


   #if USE_CALIBRATION
    int getDepthSenseColorCalibration(int devID,struct calibration * calib);
    int getDepthSenseDepthCalibration(int devID,struct calibration * calib);

    int setDepthSenseColorCalibration(int devID,struct calibration * calib);
    int setDepthSenseDepthCalibration(int devID,struct calibration * calib);
   #endif

   //Color Frame getters
   unsigned long getLastDepthSenseColorTimestamp(int devID);

   int getDepthSenseColorWidth(int devID);
   int getDepthSenseColorHeight(int devID);
   int getDepthSenseColorDataSize(int devID);
   int getDepthSenseColorChannels(int devID);
   int getDepthSenseColorBitsPerPixel(int devID);
   unsigned char * getDepthSenseColorPixels(int devID);
   double getDepthSenseColorFocalLength(int devID);
   double getDepthSenseColorPixelSize(int devID);

   //Depth Frame getters
   unsigned long getLastDepthSenseDepthTimestamp(int devID);

   int getDepthSenseDepthWidth(int devID);
   int getDepthSenseDepthHeight(int devID);
   int getDepthSenseDepthDataSize(int devID);
   int getDepthSenseDepthChannels(int devID);
   int getDepthSenseDepthBitsPerPixel(int devID);

   char * getDepthSenseDepthPixels(int devID);

   double getDepthSenseDepthFocalLength(int devID);
   double getDepthSenseDepthPixelSize(int devID);
   #endif


#ifdef __cplusplus
}
#endif

#endif // DEPTHSENSEACQUISITION_H_INCLUDED
