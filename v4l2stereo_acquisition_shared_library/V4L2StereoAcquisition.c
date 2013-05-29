#include "V4L2StereoAcquisition.h"

int startV4L2Stereo(unsigned int max_devs,char * settings)
{
 return 0;
}

int getV4L2Stereo()
{
 return 0;
} // This has to be called AFTER startV4L2Stereo

int stopV4L2Stereo()
{
 return 0;
}


int getV4L2StereoNumberOfDevices()
{
    return 0;
}

int getDevIDForV4L2StereoName(char * devName)
{
 return 0;
}

   //Basic Per Device Operations
int createV4L2StereoDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
 return 0;
}

int destroyV4L2StereoDevice(int devID)
{
 return 0;
}

int seekV4L2StereoFrame(int devID,unsigned int seekFrame)
{
 return 0;
}

int snapV4L2StereoFrames(int devID)
{
 return 0;
}

//Color Frame getters
int getV4L2StereoColorWidth(int devID)
{
 return 0;
}

int getV4L2StereoColorHeight(int devID)
{
 return 0;
}

int getV4L2StereoColorDataSize(int devID)
{
 return 0;
}

int getV4L2StereoColorChannels(int devID)
{
 return 0;
}

int getV4L2StereoColorBitsPerPixel(int devID)
{
 return 0;
}

char * getV4L2StereoColorPixels(int devID)
{
 return 0;
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
