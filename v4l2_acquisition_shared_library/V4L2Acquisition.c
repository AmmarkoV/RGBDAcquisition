
#include <stdio.h>
#include <stdlib.h>


#if BUILD_V4L2
#include "V4L2Acquisition.h"
#include "V4L2Wrapper.h"
#include "V4L2IntrinsicCalibration.h"
#include <linux/videodev2.h>

int startV4L2Module(unsigned int max_devs,char * settings)
{
 return VideoInput_InitializeLibrary(10);
}

int getV4L2()
{
 return 0;
} // This has to be called AFTER startV4L2

int stopV4L2Module()
{
 return VideoInput_DeinitializeLibrary();
}


int getV4L2NumberOfDevices()
{
 return 1;
}

int getDevIDForV4L2Name(char * devName)
{
 return 0;
}

   //Basic Per Device Operations
int createV4L2Device(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
 struct VideoFeedSettings videosettings={0};
 char BITRATE=32;
 videosettings.PixelFormat=V4L2_PIX_FMT_YUYV; BITRATE=16;// <- Common setting for UVC
 return VideoInput_OpenFeed(devID,devName,width,height,BITRATE,framerate,0,videosettings);
}

int destroyV4L2Device(int devID)
{
 return 0;
}

int seekV4L2Frame(int devID,unsigned int seekFrame)
{
 return 0;
}

int snapV4L2Frames(int devID)
{
 camera_feeds[devID].frame_decoded=0;
 camera_feeds[devID].frame=getFrame_v4l2intf(&camera_feeds[devID].v4l2_interface);
 return 0;
}

//Color Frame getters
int getV4L2ColorWidth(int devID) { return camera_feeds[devID].width; }
int getV4L2ColorHeight(int devID) { return camera_feeds[devID].height; }
int getV4L2ColorChannels(int devID) { return 3;/*camera_feeds[devID].depth;*/ }
int getV4L2ColorBitsPerPixel(int devID) { return 8; }
int getV4L2ColorDataSize(int devID) { return getV4L2ColorWidth(devID)*getV4L2ColorHeight(devID)*getV4L2ColorChannels(devID)*((unsigned int) getV4L2ColorBitsPerPixel(devID)/8); }

unsigned char * getV4L2ColorPixels(int devID)
{
 return ReturnDecodedLiveFrame(devID);
 //return camera_feeds[devID].frame;
}

double getV4L2ColorFocalLength(int devID)
{
 return 0;
}

double getV4L2ColorPixelSize(int devID)
{
 return 0;
}



int setV4L2Calibration(int devID,struct calibration * calib)
{
  camera_feeds[devID].fx=calib->intrinsic[0];
  camera_feeds[devID].fy=calib->intrinsic[4];

  camera_feeds[devID].cx=calib->intrinsic[2];
  camera_feeds[devID].cy=calib->intrinsic[5];

  camera_feeds[devID].k1=calib->k1;
  camera_feeds[devID].k2=calib->k2;
  camera_feeds[devID].p1=calib->p1;
  camera_feeds[devID].p2=calib->p2;
  camera_feeds[devID].k3=calib->k3;
  camera_feeds[devID].enableIntrinsicResectioning=1;

   camera_feeds[devID].resectionPrecalculations = (unsigned int *) malloc( camera_feeds[devID].width * camera_feeds[devID].height * sizeof(unsigned int) );
   PrecalcResectioning(
                        camera_feeds[devID].resectionPrecalculations,
                        camera_feeds[devID].width,
                        camera_feeds[devID].height,
                        camera_feeds[devID].fx,
                        camera_feeds[devID].fy,
                        camera_feeds[devID].cx,
                        camera_feeds[devID].cy,
                        camera_feeds[devID].k1,
                        camera_feeds[devID].k2,
                        camera_feeds[devID].p1,
                        camera_feeds[devID].p2,
                        camera_feeds[devID].k3
                      );
  return 1;
}



int getV4L2Calibration(int devID,struct calibration * calib)
{
  calib->intrinsic[0]=camera_feeds[devID].fx;
  calib->intrinsic[4]=camera_feeds[devID].fy;

  calib->intrinsic[2]=camera_feeds[devID].cx;
  calib->intrinsic[5]=camera_feeds[devID].cy;

  calib->k1=camera_feeds[devID].k1;
  calib->k2=camera_feeds[devID].k2;
  calib->p1=camera_feeds[devID].p1;
  calib->p2=camera_feeds[devID].p2;
  calib->k3=camera_feeds[devID].k3;
  return 1;
}

int getV4L2ColorCalibration(int devID,struct calibration * calib) { return getV4L2Calibration(devID,calib); }
int getV4L2DepthCalibration(int devID,struct calibration * calib) { return getV4L2Calibration(devID,calib); }

int setV4L2ColorCalibration(int devID,struct calibration * calib) { return setV4L2Calibration(devID,calib); }
int setV4L2DepthCalibration(int devID,struct calibration * calib) { return setV4L2Calibration(devID,calib); }




//V4L2 doesnt have any specific dDepth frame getters , so we just return null
int getV4L2DepthWidth(int devID) { return 0; }
int getV4L2DepthHeight(int devID) { return 0; }
int getV4L2DepthDataSize(int devID) { return 0; }
int getV4L2DepthChannels(int devID) { return 0; }
int getV4L2DepthBitsPerPixel(int devID) {  return 0; }
char * getV4L2DepthPixels(int devID) { return 0; }
double getV4L2DepthFocalLength(int devID) {  return 0; }
double getV4L2DepthPixelSize(int devID) { return 0; }

#else
//Null build
int startV4L2Module(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startV4L2Module called on a dummy build of V4L2Acquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_V4L2 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
