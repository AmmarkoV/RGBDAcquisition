#include <stdio.h>
#include <stdlib.h>

#include "OpenNI2Acquisition.h"

#define  BUILD_OPENNI2 1

#if BUILD_OPENNI2

#include <unistd.h>

#include <OpenNI.h>
#include <PS1080.h>

#define MAX_OPENNI2_DEVICES 16
#define ANY_OPENNI2_DEVICE MAX_OPENNI2_DEVICES*2
#define MAX_TRIES_FOR_EACH_FRAME 10


using namespace std;
using namespace openni;

Device device[MAX_OPENNI2_DEVICES];
VideoStream depth[MAX_OPENNI2_DEVICES] , color[MAX_OPENNI2_DEVICES];
VideoFrameRef depthFrame[MAX_OPENNI2_DEVICES],colorFrame[MAX_OPENNI2_DEVICES];



/*
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
                               OpenNI2 context calls
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
*/



int initializeOpenNI(unsigned int MAX_DEVICES_NEEDED)
{
   //Startup Everything!
   if(OpenNI::initialize()!=STATUS_OK)
    {
        fprintf(stderr,"Could not initialize OpenNI : %s \n",OpenNI::getExtendedError());
        return 0;
    }


    if (MAX_DEVICES_NEEDED>=MAX_OPENNI2_DEVICES)
      {
          fprintf(stderr,"\n\n\n\nPlease note that the current version of OpenNI2Aquisition  has a static limit of %u devices\n\n\n",MAX_OPENNI2_DEVICES);
      }
  return 1;
}


int closeOpenNI()
{
  fprintf(stderr,"Shutting down OpenNI\n");
  OpenNI::shutdown();
  return 1;
}



/*
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
                               OpenNI2 per device calls
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
*/



int getOpenNI2NumberOfDevices()
{
    Array<DeviceInfo> deviceInfoList;
    OpenNI::enumerateDevices(&deviceInfoList);
    return deviceInfoList.getSize();
}


const char * getURIForDeviceNumber(int deviceNumber)
{
    //This call returns a URI for a device number
    Array<DeviceInfo> deviceInfoList;
    OpenNI::enumerateDevices(&deviceInfoList);



    if (deviceInfoList.getSize()==0)
      {
        fprintf(stderr,"\n\n\n\n\n FOUND NO DEVICES , please connect an OpenNI2 compatible device \n\n\n\n\n ");
        return openni::ANY_DEVICE;
      }


    if (deviceNumber==ANY_OPENNI2_DEVICE) { return openni::ANY_DEVICE; }
      else
    if (deviceNumber>deviceInfoList.getSize())
       {
         fprintf(stderr,"Device with index %u does not exist , our total devices are %u ..\n",deviceNumber,deviceInfoList.getSize());
         fprintf(stderr,"Returning a null URI , this will probably stop the rest of the program\n");
         return 0;
       }
      else
    if (deviceInfoList.getSize()==1)
      {
         //We only have 1 device connected and we asked for deviceNumber zero , might as well return ANY_DEVICE
         //and be done with the getURIForDeviceNumber function here..
         return openni::ANY_DEVICE;
      }


    fprintf(stderr,"Found %u device(s)\n",deviceInfoList.getSize());
    int i=0;
    for (i=0; i<deviceInfoList.getSize(); i++)
     {
      if (deviceNumber==i)
      {
        fprintf(stderr,"- - - - - - - - - -\n");
        fprintf(stderr,"Selecting device #%u\n",i);
        fprintf(stderr,"- - - - - - - - - -\n");
        fprintf(stderr,"Device Index : %u \n",i);
        fprintf(stderr,"URI : %s \n",deviceInfoList[i].getUri());
        fprintf(stderr,"Name  : %s \n",deviceInfoList[i].getName());
        fprintf(stderr,"Vendor : %s \n",deviceInfoList[i].getVendor());
        fprintf(stderr,"USB vendor:productid = %04x:%04x (lsusb style) \n",deviceInfoList[i].getUsbVendorId(),deviceInfoList[i].getUsbProductId());
        fprintf(stderr,"- - - - - - - - - -\n");
        return deviceInfoList[i].getUri();
      }
    }

   return openni::ANY_DEVICE;
}

int initializeOpenNIDevice(int deviceID , Device &device , VideoStream &color , VideoStream &depth ,unsigned int width ,unsigned int height , unsigned int fps)
{

    if (device.open(getURIForDeviceNumber(deviceID)) != STATUS_OK)
    {
        fprintf(stderr,"Could not open an OpenNI device : %s \n",OpenNI::getExtendedError());
        return 0;
    }

  //device.setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);

if (device.getSensorInfo(SENSOR_DEPTH)  != NULL)
    {
        Status rc = depth.create(device, SENSOR_DEPTH);
        if (rc == STATUS_OK)
        {
            VideoMode depthMode = depth.getVideoMode();
            depthMode.setResolution(width,height);
            depthMode.setFps(fps);
            Status rc = depth.setVideoMode(depthMode);
            if (rc != STATUS_OK) { fprintf(stderr,"Error getting color at video mode requested %u x %u @ %u fps\n%s\n",width,height,fps,OpenNI::getExtendedError()); }

            if(depth.start()!= STATUS_OK)
            {
                fprintf(stderr,"Couldn't start the color stream: %s \n",OpenNI::getExtendedError());
                return 0;
            }
        }
        else
        {
            fprintf(stderr,"Couldn't create depth stream: %s \n",OpenNI::getExtendedError());
            return 0;
        }
    }

    if (device.getSensorInfo(SENSOR_COLOR) != NULL)
    {
        Status rc = color.create(device, SENSOR_COLOR);
        if (rc == STATUS_OK)
        {
            VideoMode colorMode = color.getVideoMode();
            colorMode.setResolution(width,height);
            colorMode.setFps(fps);
            Status rc = color.setVideoMode(colorMode);
            if (rc != STATUS_OK) { fprintf(stderr,"Error getting depth at video mode requested %u x %u @ %u fps\n%s\n",width,height,fps,OpenNI::getExtendedError()); }

            if(color.start() != STATUS_OK)
            {
                fprintf(stderr,"Couldn't start the color stream: %s \n",OpenNI::getExtendedError());
                return 0;
            }
        }
        else
        {
            fprintf(stderr,"Couldn't create depth stream: %s \n",OpenNI::getExtendedError());
            OpenNI::getExtendedError();
            return 0;
        }
    }

    depth.setMirroringEnabled (false);
    color.setMirroringEnabled (false);


    fprintf(stdout,"Device Initialization Requested %u x %u @ %u fps \n",width,height,fps);
   return 1;
}




int closeOpenNIDevice(Device &device , VideoStream &color , VideoStream &depth)
{
    fprintf(stderr,"Stopping depth and color streams\n");
    depth.stop();
    color.stop();
    depth.destroy();
    color.destroy();
    device.close();
    return 1;
}

/*
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
                               OpenNI2 read frames
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
*/


int readFrameBlocking(VideoStream &stream,VideoFrameRef &frame , unsigned int max_tries)
{
  unsigned int tries_for_frame = 0 ;
  while (  tries_for_frame < max_tries  )
          {
            stream.readFrame(&frame);
	        if (frame.isValid()) { return 1; }
	        ++tries_for_frame;
          }

  if (!frame.isValid()) { fprintf(stderr,"Could not get a valid frame even after %u tries \n",max_tries); return 0; }
  return (tries_for_frame<max_tries);
}

int readOpenNiColorAndDepth(VideoStream &color , VideoStream &depth,VideoFrameRef &colorFrame,VideoFrameRef &depthFrame)
{
    readFrameBlocking(depth,depthFrame,MAX_TRIES_FOR_EACH_FRAME); // depth.readFrame(&depthFrame);
    readFrameBlocking(color,colorFrame,MAX_TRIES_FOR_EACH_FRAME); // color.readFrame(&colorFrame);

    if(depthFrame.isValid() && colorFrame.isValid()) { return 1; }

    fprintf(stderr,"Depth And Color frames are wrong!\n");
    return 0;
}
/*
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
*/







/*


   EXTERNAL EXPOSED FUNCTIONS

*/
int mapOpenNI2DepthToRGB(int devID)
{
  device[devID].setImageRegistrationMode(IMAGE_REGISTRATION_DEPTH_TO_COLOR);
  return 1;
}


int mapOpenNI2RGBToDepth(int devID)
{
  return 0;
  //device[devID].setImageRegistrationMode(IMAGE_REGISTRATION_COLOR_TO_DEPTH);
  return 1;
}




int startOpenNI2Module(unsigned int max_devs)
{
    return initializeOpenNI(max_devs);
}

int stopOpenNI2Module()
{
    return closeOpenNI();
}


int snapOpenNI2Frames(int devID)
{
  return readOpenNiColorAndDepth(color[devID],depth[devID],colorFrame[devID],depthFrame[devID]);
}

int createOpenNI2Device(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
  {
    if (! initializeOpenNIDevice(devID,device[devID],color[devID],depth[devID],width,height,framerate) )
     {
         fprintf(stderr,"Could not initialize device with ID %u \n",devID);
         return 0;
     }


     //Snapping an initial frame to populate Image Sizes , etc..
    if ( ! snapOpenNI2Frames(devID) )
    {
     //Could not snap the initial frame
      fprintf(stderr,"Could not snap an initial frame off the OpenNI2 device\n");
      //return 0; although we have initializedOpenNI without an error what should we return here..
      //we will return 1!
    } else
    {

     //Frame Width/Height /Focal length , etc should be ok..
     fprintf(stdout,"Color Frames : %u x %u , channels %u , bitsperpixel %u \n",getOpenNI2ColorWidth(devID), getOpenNI2ColorHeight(devID) , getOpenNI2ColorChannels(devID) , getOpenNI2ColorBitsPerPixel(devID));
     fprintf(stdout,"Color Focal Length : %0.2f\n",getOpenNI2ColorFocalLength(devID));
     fprintf(stdout,"Color Pixel Size : %0.2f\n",getOpenNI2ColorPixelSize(devID));

     fprintf(stdout,"Depth Frames : %u x %u , channels %u , bitsperpixel %u \n",getOpenNI2DepthWidth(devID), getOpenNI2DepthHeight(devID), getOpenNI2DepthChannels(devID) , getOpenNI2DepthBitsPerPixel(devID));
     fprintf(stdout,"Depth Focal Length : %0.2f\n",getOpenNI2DepthFocalLength(devID));
     fprintf(stdout,"Depth Pixel Size : %0.2f\n",getOpenNI2DepthPixelSize(devID));
    }
    return 1;
  }


 int destroyOpenNI2Device(int devID)
 {
     return closeOpenNIDevice(device[devID] , color[devID] , depth[devID]);
 }



//COLOR FRAMES
int getOpenNI2ColorWidth(int devID)
{
   return colorFrame[devID].getWidth();
}
int getOpenNI2ColorHeight(int devID)
{
   return colorFrame[devID].getHeight();
}
int getOpenNI2ColorDataSize(int devID)
{
    return colorFrame[devID].getDataSize();
}
int getOpenNI2ColorChannels(int devID)
{
    return 3;
}
int getOpenNI2ColorBitsPerPixel(int devID)
{
    return 8;
}
char * getOpenNI2ColorPixels(int devID)
{
   return (char *)colorFrame[devID].getData();
}


double getOpenNI2ColorFocalLength(int devID)
{
    int zpd=0;
    color[devID].getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE,&zpd);
    return (double) zpd;
}

double getOpenNI2ColorPixelSize(int devID)
{
    double zpps=0.0;
    color[devID].getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE,&zpps);
    return (double) zpps;
}




//DEPTH FRAMES
int getOpenNI2DepthWidth(int devID)
{
   return depthFrame[devID].getWidth();
}
int getOpenNI2DepthHeight(int devID)
{
   return depthFrame[devID].getHeight();
}
int getOpenNI2DepthDataSize(int devID)
{
    return depthFrame[devID].getDataSize();
}
int getOpenNI2DepthChannels(int devID)
{
    return 1;
}
int getOpenNI2DepthBitsPerPixel(int devID)
{
    return 16;
}
short * getOpenNI2DepthPixels(int devID)
{
   return (short *) depthFrame[devID].getData();
}


double getOpenNI2DepthFocalLength(int devID)
{
    int zpd=0;
    depth[devID].getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_DISTANCE,&zpd);
    return (double) zpd;
}

double getOpenNI2DepthPixelSize(int devID)
{
    double zpps=0.0;
    depth[devID].getProperty(XN_STREAM_PROPERTY_ZERO_PLANE_PIXEL_SIZE,&zpps);
    return (double) zpps;
}


#else

//Null build
int startOpenNI2Module(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startOpenNI2Module called on a dummy build of OpenNI2Acquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_OPENNI2 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
