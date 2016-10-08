
#include <stdio.h>
#include <stdlib.h>

#include "RealsenseAcquisition.h"

//#define BUILD_REALSENSE 1

#if BUILD_REALSENSE
#include "../3dparty/librealsense/include/librealsense/rs.h"
rs_context * ctx = 0;
rs_device * dev = 0;

rs_error * e = 0;

unsigned int frameCount=0;
void check_error()
{
    if(e)
    {
        printf("rs_error was raised when calling %s(%s):\n", rs_get_failed_function(e), rs_get_failed_args(e));
        printf("    %s\n", rs_get_error_message(e));
        exit(EXIT_FAILURE);
    }
}


int startRealsenseModule(unsigned int max_devs,char * settings)
{
 /* Create a context object. This object owns the handles to all connected realsense devices. */
    ctx = rs_create_context(RS_API_VERSION, &e);
    check_error();
    printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));
    check_error();
    if(rs_get_device_count(ctx, &e) == 0) return EXIT_FAILURE;

  return 1;
}

int createRealsenseDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
    /* This tutorial will access only a single device, but it is trivial to extend to multiple devices */
    dev = rs_get_device(ctx, 0, &e);
    check_error();
    printf("\nUsing device 0, an %s\n", rs_get_device_name(dev, &e));
    check_error();
    printf("    Serial number: %s\n", rs_get_device_serial(dev, &e));
    check_error();
    printf("    Firmware version: %s\n", rs_get_device_firmware_version(dev, &e));
    check_error();

    /* Configure all streams to run at VGA resolution at 60 frames per second */
    rs_enable_stream(dev, RS_STREAM_DEPTH, 640, 480, RS_FORMAT_Z16, 60, &e);
    check_error();
    rs_enable_stream(dev, RS_STREAM_COLOR, 640, 480, RS_FORMAT_RGB8, 60, &e);
    check_error();
    rs_enable_stream(dev, RS_STREAM_INFRARED, 640, 480, RS_FORMAT_Y8, 60, &e);
    check_error();
    rs_enable_stream(dev, RS_STREAM_INFRARED2, 640, 480, RS_FORMAT_Y8, 60, NULL); /* Pass NULL to ignore errors */
    rs_start_device(dev, &e);
    check_error();
  return 1;
}

int stopRealsenseModule() { return 1; }

int getRealsenseNumberOfDevices()  {  return rs_get_device_count(ctx, &e); }


int mapRealsenseDepthToRGB(int devID)
{
   return 1;
}

int seekRealsenseFrame(int devID,unsigned int seekFrame)
{
  return 0;
}



int getTotalRealsenseFrameNumber(int devID)
{
 return 0;
}


int getCurrentRealsenseFrameNumber(int devID)
{
 return frameCount;
}


int snapRealsenseFrames(int devID)
{
  rs_wait_for_frames(dev, &e);
  check_error();
  ++frameCount;
  return 1;
}

//Color Frame getters
int getRealsenseColorWidth(int devID) { return 640; }
int getRealsenseColorHeight(int devID) { return 480; }
int getRealsenseColorDataSize(int devID) { return getRealsenseColorWidth(devID)*getRealsenseColorHeight(devID)*3; }
int getRealsenseColorChannels(int devID) { return 3; }
int getRealsenseColorBitsPerPixel(int devID) { return 8; }

char * getRealsenseColorPixels(int devID)
{
  return rs_get_frame_data(dev, RS_STREAM_COLOR, &e);
}


//Depth Frame getters
int getRealsenseDepthWidth(int devID) { return 640; }
int getRealsenseDepthHeight(int devID) { return 480; }
int getRealsenseDepthDataSize(int devID) { return getRealsenseDepthWidth(devID)*getRealsenseDepthHeight(devID); }
int getRealsenseDepthChannels(int devID) { return 1; }
int getRealsenseDepthBitsPerPixel(int devID) { return 16; }

char * getRealsenseDepthPixels(int devID)
{
  return rs_get_frame_data(dev, RS_STREAM_DEPTH, &e);
}


#else
//Null build
int startRealsenseModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startRealsenseModule called on a dummy build of RealsenseAcquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_FREENECT 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
