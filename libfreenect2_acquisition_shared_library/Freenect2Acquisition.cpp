
#include <stdio.h>
#include <stdlib.h>

#include "Freenect2Acquisition.h"

#if BUILD_FREENECT2
#include "../3dparty/libfreenect/include/libfreenect.h"
#include "../3dparty/libfreenect/wrappers/c_sync/libfreenect_sync.h"

#define MAX_DEVS 16

int rgb_mode[MAX_DEVS]={FREENECT_VIDEO_RGB};
int depth_mode[MAX_DEVS]={FREENECT_DEPTH_11BIT};

int startFreenect2Module(unsigned int max_devs,char * settings)
{
  uint32_t ts;
  char * rgb, * depth;
  fprintf(stderr,"Please hang on while starting Freenect2 module.. \n");
  int ret = freenect_sync_get_video((void**)&rgb, &ts, 0 , FREENECT_VIDEO_RGB);
  if (ret < 0) { fprintf(stderr,"There doesnt seem to exist a Freenect2 compatible device with index 0\n"); return 0; }

  freenect_sync_get_video((void**)&rgb, &ts, 0 , FREENECT_VIDEO_RGB);
  freenect_sync_get_depth((void**)&depth, &ts, 0 ,FREENECT_DEPTH_11BIT);
  return 1;
}

int createFreenect2Device(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
  uint32_t ts;
  char * rgb;//, * depth;
  int ret = freenect_sync_get_video((void**)&rgb, &ts, 0 , FREENECT_VIDEO_RGB);
  if (ret < 0) { fprintf(stderr,"There doesnt seem to exist a Freenect2 compatible device with index 0\n"); return 0; }
  return 1;
}

int stopFreenect2Module() { return 1; }

int getFreenect2NumberOfDevices()  { fprintf(stderr,"New getFreenect2NumberOfDevices is a stub it always returns 1"); return 1; }


int mapFreenect2DepthToRGB(int devID)
{
   depth_mode[devID]=FREENECT_DEPTH_REGISTERED;
   return 1;
}

int seekFreenect2Frame(int devID,unsigned int seekFrame)
{
  return 0;
}

int snapFreenect2Frames(int devID)
{
  return 1;
}

//Color Frame getters
int getFreenect2ColorWidth(int devID) { return 640; }
int getFreenect2ColorHeight(int devID) { return 480; }
int getFreenect2ColorDataSize(int devID) { return getFreenect2ColorWidth(devID)*getFreenect2ColorHeight(devID)*3; }
int getFreenect2ColorChannels(int devID) { return 3; }
int getFreenect2ColorBitsPerPixel(int devID) { return 8; }

char * getFreenect2ColorPixels(int devID)
{
  uint32_t ts;
  char * rgb;
  int ret = freenect_sync_get_video((void**)&rgb, &ts, devID , rgb_mode[devID]);
  if (ret < 0) { fprintf(stderr,"There doesnt seem to exist a Freenect2 compatible device with index %u\n",devID); return 0; }
  return rgb;
}


//Depth Frame getters
int getFreenect2DepthWidth(int devID) { return 640; }
int getFreenect2DepthHeight(int devID) { return 480; }
int getFreenect2DepthDataSize(int devID) { return getFreenect2DepthWidth(devID)*getFreenect2DepthHeight(devID); }
int getFreenect2DepthChannels(int devID) { return 1; }
int getFreenect2DepthBitsPerPixel(int devID) { return 16; }

char * getFreenect2DepthPixels(int devID)
{
  uint32_t ts;
  char * depth;
  int ret = freenect_sync_get_depth((void**)&depth, &ts, devID , depth_mode[devID]);
  if (ret < 0) { fprintf(stderr,"There doesnt seem to exist a Freenect2 compatible device with index %u\n",devID);  return 0; }
  return depth;
}


#else
//Null build
int startFreenect2Module(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startFreenect2Module called on a dummy build of Freenect2Acquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_FREENECT 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
