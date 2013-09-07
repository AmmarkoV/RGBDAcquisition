
#include <stdio.h>
#include <stdlib.h>

#include "../acquisition/acquisition_setup.h"

#include "../3dparty/libfreenect/include/libfreenect.h"
#include "../3dparty/libfreenect/wrappers/c_sync/libfreenect_sync.h"

#define MAX_DEVS 16

int rgb_mode[MAX_DEVS]={FREENECT_VIDEO_RGB};
int depth_mode[MAX_DEVS]={FREENECT_DEPTH_11BIT};

int mapFreenectDepthToRGB(int devID)
{
   depth_mode[devID]=FREENECT_DEPTH_REGISTERED;
   return 1;
}


int startFreenectModule(unsigned int max_devs,char * settings)
{
  uint32_t ts;
  char * rgb, * depth;
  fprintf(stderr,"Please hang on while starting Freenect module.. \n");
  int ret = freenect_sync_get_video((void**)&rgb, &ts, 0 , FREENECT_VIDEO_RGB);
  if (ret < 0) { fprintf(stderr,"There doesnt seem to exist a Freenect compatible device with index 0\n"); return 0; }

  freenect_sync_get_video((void**)&rgb, &ts, 0 , FREENECT_VIDEO_RGB);
  freenect_sync_get_depth((void**)&depth, &ts, 0 ,FREENECT_DEPTH_11BIT);

  return 1;
}

int stopFreenectModule() { return 1; }

int getFreenectNumberOfDevices()  { fprintf(stderr,"New getFreenectNumberOfDevices is a stub it always returns 1"); return 1; }


int seekFreenectFrame(int devID,unsigned int seekFrame)
{
  return 0;
}

int snapFreenectFrames(int devID)
{
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
