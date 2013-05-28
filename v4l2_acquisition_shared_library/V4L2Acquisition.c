#include "V4L2Acquisition.h"

int startV4L2(unsigned int max_devs,char * settings)
{
 return 0;
}

int getV4L2()
{
 return 0;
} // This has to be called AFTER startV4L2

int stopV4L2()
{
 return 0;
}

int getDevIDForV4L2Name(char * devName)
{
 return 0;
}

   //Basic Per Device Operations
int createV4L2Device(int devID,unsigned int width,unsigned int height,unsigned int framerate)
{
 return 0;
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
 return 0;
}

//Color Frame getters
int getV4L2ColorWidth(int devID)
{
 return 0;
}

int getV4L2ColorHeight(int devID)
{
 return 0;
}

int getV4L2ColorDataSize(int devID)
{
 return 0;
}

int getV4L2ColorChannels(int devID)
{
 return 0;
}

int getV4L2ColorBitsPerPixel(int devID)
{
 return 0;
}

char * getV4L2ColorPixels(int devID)
{
 return 0;
}

double getV4L2ColorFocalLength(int devID)
{
 return 0;
}

double getV4L2ColorPixelSize(int devID)
{
 return 0;
}

   //Depth Frame getters
int getV4L2DepthWidth(int devID)
{
 return 0;
}

int getV4L2DepthHeight(int devID)
{
 return 0;
}

int getV4L2DepthDataSize(int devID)
{
 return 0;
}

int getV4L2DepthChannels(int devID)
{
 return 0;
}

int getV4L2DepthBitsPerPixel(int devID)
{
 return 0;
}

char * getV4L2DepthPixels(int devID)
{
 return 0;
}

double getV4L2DepthFocalLength(int devID)
{
 return 0;
}

double getV4L2DepthPixelSize(int devID)
{
 return 0;
}
