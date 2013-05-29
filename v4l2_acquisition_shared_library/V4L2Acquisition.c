#include "V4L2Acquisition.h"
#include "V4L2Wrapper.h"
#include <linux/videodev2.h>

int startV4L2(unsigned int max_devs,char * settings)
{
 return VideoInput_InitializeLibrary(10);
}

int getV4L2()
{
 return 0;
} // This has to be called AFTER startV4L2

int stopV4L2()
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
 camera_feeds[devID].frame=getFrame_v4l2intf(&camera_feeds[devID].v4l2_interface);
 return 0;
}

//Color Frame getters
int getV4L2ColorWidth(int devID)
{
 return camera_feeds[devID].width;
}

int getV4L2ColorHeight(int devID)
{
 return camera_feeds[devID].height;
}

int getV4L2ColorChannels(int devID)
{
 return 3;//camera_feeds[devID].depth;
}

int getV4L2ColorBitsPerPixel(int devID)
{
 return 8;
}

int getV4L2ColorDataSize(int devID)
{
 return getV4L2ColorWidth(devID)*getV4L2ColorHeight(devID)*getV4L2ColorChannels(devID)*((unsigned int) getV4L2ColorBitsPerPixel(devID)/8);
}

char * getV4L2ColorPixels(int devID)
{
 camera_feeds[devID].frame=getFrame_v4l2intf(&camera_feeds[devID].v4l2_interface);
 return camera_feeds[devID].frame;
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
