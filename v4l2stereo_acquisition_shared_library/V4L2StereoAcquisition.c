#include "V4L2StereoAcquisition.h"
#include "../acquisition/Acquisition.h"
#include "../v4l2_acquisition_shared_library/V4L2Acquisition.h"

int startV4L2Stereo(unsigned int max_devs,char * settings)
{
 return startV4L2(max_devs,settings);
}

int getV4L2Stereo()
{
 return 0;
} // This has to be called AFTER startV4L2Stereo

int stopV4L2Stereo()
{
 return stopV4L2();
}


int getV4L2StereoNumberOfDevices()
{
    return 1;
}

int getDevIDForV4L2StereoName(char * devName)
{
 return 0;
}

   //Basic Per Device Operations
int createV4L2StereoDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
 createV4L2Device(0,"/dev/video1",width,height,framerate);
 createV4L2Device(1,"/dev/video2",width,height,framerate);

 return 1;
}

int destroyV4L2StereoDevice(int devID)
{
 destroyV4L2Device(0);
 destroyV4L2Device(1);
 return 0;
}

int seekV4L2StereoFrame(int devID,unsigned int seekFrame)
{
 return 0;
}

int snapV4L2StereoFrames(int devID)
{
 snapV4L2Frames(0);
 snapV4L2Frames(1);
 return 0;
}

//Color Frame getters
int getV4L2StereoColorWidth(int devID) { return getV4L2ColorWidth(devID); }
int getV4L2StereoColorHeight(int devID) { return getV4L2ColorHeight(devID); }
int getV4L2StereoColorDataSize(int devID) { return getV4L2ColorDataSize(devID); }
int getV4L2StereoColorChannels(int devID) {  return getV4L2ColorChannels(devID); }
int getV4L2StereoColorBitsPerPixel(int devID) {  return getV4L2ColorBitsPerPixel(devID); }

char * getV4L2StereoColorPixels(int devID)
{
 return getV4L2StereoColorPixels(devID);
}

char * getV4L2StereoColorPixelsLeft(int devID)
{
 return getV4L2StereoColorPixels(devID);
}

char * getV4L2StereoColorPixelsRight(int devID)
{
 return getV4L2StereoColorPixels(devID+1);
}

double getV4L2StereoColorFocalLength(int devID)
{
 return 0;
}

double getV4L2StereoColorPixelSize(int devID)
{
 return 0;
}

   //Depth Frame getters
int getV4L2StereoDepthWidth(int devID)
{
 return 0;
}

int getV4L2StereoDepthHeight(int devID)
{
 return 0;
}

int getV4L2StereoDepthDataSize(int devID)
{
 return 0;
}

int getV4L2StereoDepthChannels(int devID)
{
 return 0;
}

int getV4L2StereoDepthBitsPerPixel(int devID)
{
 return 0;
}

char * getV4L2StereoDepthPixels(int devID)
{
 return 0;
}

double getV4L2StereoDepthFocalLength(int devID)
{
 return 0;
}

double getV4L2StereoDepthPixelSize(int devID)
{
 return 0;
}
