
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "FreenectAcquisition.h"

#define BUILD_FREENECT 1
#define USE_CALIBRATION 1

#if BUILD_FREENECT
#include "../3dparty/libfreenect/include/libfreenect.h"
#include "../3dparty/libfreenect/wrappers/c_sync/libfreenect_sync.h"

#define MAX_DEVS 16


#if USE_CALIBRATION
 struct calibration calibRGB[MAX_DEVS];
 struct calibration calibDepth[MAX_DEVS];
#endif


int rgb_mode[MAX_DEVS]={FREENECT_VIDEO_RGB};
int depth_mode[MAX_DEVS]={FREENECT_DEPTH_11BIT};
int numberOfFramesSnapped[MAX_DEVS]={0};




double getFreenectColorPixelSize(int devID)
{
 #warning "Freenect implementation returns a dummy pixel size.."
 double zpps=0.1052;
 return (double) zpps;
}

double getFreenectColorFocalLength(int devID)
{
  unsigned int width=getFreenectColorWidth(devID);
  unsigned int height=getFreenectColorHeight(devID);
  double zpps=getFreenectColorPixelSize(devID);
  if ( (width==640)&&(height==480) ) { return (double) 531.15 * zpps; } else
  if ( (width==320)&&(height==240) ) { return (double) 285.63 * zpps; } else
  return (double) 120.0;
}


double getFreenectDepthPixelSize(int devID)
{
 double zpps=0.1052;
 return (double) zpps;
}

double getFreenectDepthFocalLength(int devID)
{
  unsigned int width=getFreenectDepthWidth(devID);
  unsigned int height=getFreenectDepthHeight(devID);
  double zpps=getFreenectDepthPixelSize(devID);
  if ( (width==640)&&(height==480) ) { return (double) 531.15 * zpps; } else
  if ( (width==320)&&(height==240) ) { return (double) 285.63 * zpps; } else
  return (double) 120.0;
}









int startFreenectModule(unsigned int max_devs,char * settings)
{
  fprintf(stderr,"Please note that the calibration data coming out of the freenect module is dummy..\n");
  uint32_t ts;
  char * rgb, * depth;
  fprintf(stderr,"Please hang on while starting Freenect module.. \n");
  int ret = freenect_sync_get_video((void**)&rgb, &ts, 0 , FREENECT_VIDEO_RGB);
  if (ret < 0) { fprintf(stderr,"There doesnt seem to exist a Freenect compatible device with index 0\n"); return 0; }

  freenect_sync_get_video((void**)&rgb, &ts, 0 , FREENECT_VIDEO_RGB);
  freenect_sync_get_depth((void**)&depth, &ts, 0 ,FREENECT_DEPTH_11BIT);
  return 1;
}

int createFreenectDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
  uint32_t ts;
  char * rgb;//, * depth;


  int ret = freenect_sync_get_video((void**)&rgb, &ts, 0 , FREENECT_VIDEO_RGB);
  if (ret < 0) { fprintf(stderr,"There doesnt seem to exist a Freenect compatible device with index 0\n"); return 0; }

  freenect_sync_set_tilt_degs(0,0);

  #if USE_CALIBRATION
     //Populate our calibration data ( if we have them )
     FocalLengthAndPixelSizeToCalibration(getFreenectColorFocalLength(devID),
                                          getFreenectColorPixelSize(devID),
                                          getFreenectColorWidth(devID),
                                          getFreenectColorHeight(devID),
                                          &calibRGB[devID]);

     FocalLengthAndPixelSizeToCalibration(getFreenectDepthFocalLength(devID),
                                          getFreenectDepthPixelSize(devID),
                                          getFreenectDepthWidth(devID),
                                          getFreenectDepthHeight(devID),
                                          &calibDepth[devID]);
  #endif

 return 1;
}

int stopFreenectModule() { freenect_sync_stop(); return 1; }

int getFreenectNumberOfDevices()  { fprintf(stderr,"New getFreenectNumberOfDevices is a stub it always returns 1"); return 1; }


int mapFreenectDepthToRGB(int devID)
{
   depth_mode[devID]=FREENECT_DEPTH_REGISTERED;
   return 1;
}

int seekFreenectFrame(int devID,unsigned int seekFrame)
{
  return 0;
}

int snapFreenectFrames(int devID)
{
  numberOfFramesSnapped[devID]+=1;
  return 1;
}

//Color Frame getters
int getFreenectColorWidth(int devID) { return 640; }
int getFreenectColorHeight(int devID) { return 480; }
int getFreenectColorDataSize(int devID) { return getFreenectColorWidth(devID)*getFreenectColorHeight(devID)*3; }
int getFreenectColorChannels(int devID) { return 3; }
int getFreenectColorBitsPerPixel(int devID) { return 8; }

char * getFreenectColorPixels(int devID)
{
  uint32_t ts;
  char * rgb;
  int ret = freenect_sync_get_video((void**)&rgb, &ts, devID , rgb_mode[devID]);
  if (ret < 0) { fprintf(stderr,"There doesnt seem to exist a Freenect compatible device with index %u\n",devID); return 0; }
  return rgb;
}


//Depth Frame getters
int getFreenectDepthWidth(int devID) { return 640; }
int getFreenectDepthHeight(int devID) { return 480; }
int getFreenectDepthDataSize(int devID) { return getFreenectDepthWidth(devID)*getFreenectDepthHeight(devID); }
int getFreenectDepthChannels(int devID) { return 1; }
int getFreenectDepthBitsPerPixel(int devID) { return 16; }

char * getFreenectDepthPixels(int devID)
{
  uint32_t ts;
  char * depth;
  int ret = freenect_sync_get_depth((void**)&depth, &ts, devID , depth_mode[devID]);
  if (ret < 0) { fprintf(stderr,"There doesnt seem to exist a Freenect compatible device with index %u\n",devID);  return 0; }
  return depth;
}

int getTotalFreenectFrameNumber(int devID)
{
  //This is a live device so we don't know how many frames there will be..
  return 0;
}

int getCurrentFreenectFrameNumber(int devID)
{
  return numberOfFramesSnapped[devID];
}


#if USE_CALIBRATION
int getFreenectColorCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &calibRGB[devID],sizeof(struct calibration));
    return 1;
}

int getFreenectDepthCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &calibDepth[devID],sizeof(struct calibration));
    return 1;
}


int setFreenectColorCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) &calibRGB[devID] , (void*) calib,sizeof(struct calibration));
    return 1;
}

int setFreenectDepthCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) &calibDepth[devID] , (void*) calib,sizeof(struct calibration));
    return 1;
}
#endif


#else
//Null build
int startFreenectModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startFreenectModule called on a dummy build of FreenectAcquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_FREENECT 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
