
#include <stdio.h>
#include <stdlib.h>

#include "RealsenseAcquisition.h"

#define MAX_REALSENSE_DEVICES 12
#define BUILD_REALSENSE 1

#if BUILD_REALSENSE
#include "../3dparty/librealsense/include/librealsense/rs.h"

rs_error * e;
rs_context * ctx;

struct realsenseDeviceWrapper
{
 rs_device * dev;
 int colorStreamToUse;
 int depthStreamToUse;
 unsigned int frameCount;
 unsigned int hasInit;
};

struct realsenseDeviceWrapper device[MAX_REALSENSE_DEVICES]={0};

int check_error()
{
    if(e)
    {
        fprintf(stderr,"rs_error was raised when calling %s(%s):\n", rs_get_failed_function(e), rs_get_failed_args(e));
        fprintf(stderr,"    %s\n", rs_get_error_message(e));
        return 1;
        //exit(EXIT_FAILURE);
    }
    return 0;
}


int startRealsenseModule(unsigned int max_devs,char * settings)
{
 /* Create a context object. This object owns the handles to all connected realsense devices. */
    ctx = rs_create_context(RS_API_VERSION, &e);
    if ( check_error() )
    {
      fprintf(stderr,"Cannot create a device context for Realsense devices \n");
      return 0;
    }

    printf("There are %d connected RealSense devices.\n", rs_get_device_count(ctx, &e));  check_error();
    if(rs_get_device_count(ctx, &e) == 0) return 0;


    unsigned int i=0;
    for (i=0; i<MAX_REALSENSE_DEVICES; i++)
    {
         device[i].colorStreamToUse=RS_STREAM_COLOR;
         device[i].depthStreamToUse=RS_STREAM_DEPTH_ALIGNED_TO_COLOR;
    }

  return 1;
}

int createRealsenseDevice(int devID,char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
    /* This call will access only a single device, based on the devID identifier..*/
    device[devID].dev = rs_get_device(ctx, devID , &e);
    if (check_error())
    {
      fprintf(stderr,"Cannot get a device context for device %u \n",devID);
      return 0;
    }

    printf("\nUsing device 0, an %s\n", rs_get_device_name(device[devID].dev, &e));                       check_error();
    printf("    Serial number: %s\n", rs_get_device_serial(device[devID].dev, &e));                       check_error();
    printf("    Firmware version: %s\n", rs_get_device_firmware_version(device[devID].dev, &e));          check_error();

    /* Configure all streams to run at VGA resolution at 60 frames per second */
    rs_enable_stream(device[devID].dev, RS_STREAM_DEPTH, width, height, RS_FORMAT_Z16, framerate, &e);    check_error();
    rs_enable_stream(device[devID].dev, RS_STREAM_COLOR, width, height, RS_FORMAT_RGB8, framerate, &e);   check_error();
    rs_start_device(device[devID].dev, &e);                                                               check_error();

    device[devID].colorStreamToUse=RS_STREAM_COLOR;
    device[devID].depthStreamToUse=RS_STREAM_DEPTH_ALIGNED_TO_COLOR;
    device[devID].frameCount=0;
    device[devID].hasInit=1;
  return 1;
}

int destroyRealsenseDevice(int devID)
{
  device[devID].hasInit=0;
  rs_stop_device(device[devID].dev, &e);                          check_error();
  rs_disable_stream(device[devID].dev, RS_STREAM_DEPTH, &e);    check_error();
  rs_disable_stream(device[devID].dev, RS_STREAM_COLOR, &e);    check_error();
  return 1;
}

int stopRealsenseModule()          { rs_delete_context(ctx, &e); check_error(); return 1; }

int getRealsenseNumberOfDevices()  {  return rs_get_device_count(ctx, &e); }

int mapRealsenseDepthToRGB(int devID)
{
  device[devID].colorStreamToUse=RS_STREAM_COLOR;
  device[devID].depthStreamToUse=RS_STREAM_DEPTH_ALIGNED_TO_COLOR;
  return 1;
}
int mapRealsenseRGBToDepth(int devID)
{
   device[devID].colorStreamToUse=RS_STREAM_COLOR_ALIGNED_TO_DEPTH;
   device[devID].depthStreamToUse=RS_STREAM_DEPTH;
   return 1;
}

int seekRealsenseFrame(int devID,unsigned int seekFrame)  { return 0; }
int getTotalRealsenseFrameNumber(int devID)               { return 0; }
int getCurrentRealsenseFrameNumber(int devID)             { return device[devID].frameCount; }



int snapRealsenseFrames(int devID)
{
  rs_wait_for_frames(device[devID].dev, &e);
  check_error();
  ++device[devID].frameCount;
  return 1;
}

//Color Frame getters
int getRealsenseColorWidth(int devID)        { return rs_get_stream_width( device[devID].dev,   device[devID].colorStreamToUse  , &e ); }
int getRealsenseColorHeight(int devID)       { return rs_get_stream_height(device[devID].dev,   device[devID].colorStreamToUse  , &e ); }
int getRealsenseColorDataSize(int devID)     { return getRealsenseColorWidth(devID)*getRealsenseColorHeight(devID)*3; }
int getRealsenseColorChannels(int devID)     { return 3; }
int getRealsenseColorBitsPerPixel(int devID) { return 8; }

char * getRealsenseColorPixels(int devID)    {  return rs_get_frame_data(device[devID].dev, device[devID].colorStreamToUse, &e); }


//Depth Frame getters
int getRealsenseDepthWidth(int devID)        { return rs_get_stream_width( device[devID].dev,   device[devID].depthStreamToUse  , &e );  }
int getRealsenseDepthHeight(int devID)       { return rs_get_stream_height(device[devID].dev,   device[devID].depthStreamToUse  , &e );  }
int getRealsenseDepthDataSize(int devID)     { return getRealsenseDepthWidth(devID)*getRealsenseDepthHeight(devID); }
int getRealsenseDepthChannels(int devID)     { return 1; }
int getRealsenseDepthBitsPerPixel(int devID) { return 16; }

char * getRealsenseDepthPixels(int devID)    { return rs_get_frame_data(device[devID].dev, device[devID].depthStreamToUse , &e); }


#define USE_CALIBRATION 1

#if USE_CALIBRATION
int getRealsenseGenericCalibration(int devID,int streamID,struct calibration * calib)
{
    fprintf(stderr,"For some reason rs_get_stream_intrinsics segfaults on call , so Calibration is disabled.. \n");
    return 0;

    calib->intrinsicParametersSet=0;
    if (!device[devID].hasInit)
    {
          fprintf(stderr,"A Calibration was asked before setting up the streams .. \n");
          return 0;
    }

     snapRealsenseFrames(devID);
     getRealsenseColorPixels(devID) ;
     getRealsenseDepthPixels(devID) ;


    fprintf(stderr,"rs_intrinsics is now declared \n");
    rs_intrinsics intrin;
    fprintf(stderr,"rs_get_stream_intrinsics will now be called\n");
    rs_get_stream_intrinsics( device[devID].dev, streamID , &intrin, &e);

    if ( !check_error() )
    {
    fprintf(stderr,"Survived Intrinsics");

    calib->intrinsicParametersSet=1;
    calib->k1=intrin.coeffs[0]; calib->k2=intrin.coeffs[1];
    calib->p1=intrin.coeffs[2]; calib->p2=intrin.coeffs[3];
    calib->k3=intrin.coeffs[4];

    //clear first
    calib->intrinsic[0]=0.0;  calib->intrinsic[1]=0.0;  calib->intrinsic[2]=0.0;
    calib->intrinsic[3]=0.0;  calib->intrinsic[4]=0.0;  calib->intrinsic[5]=0.0;
    calib->intrinsic[6]=0.0;  calib->intrinsic[7]=0.0;  calib->intrinsic[8]=1.0;
    //fill after..
    calib->intrinsic[CALIB_INTR_FX] = (double) intrin.fx/intrin.ppx;
    calib->intrinsic[CALIB_INTR_FY] = (double) intrin.fy/intrin.ppy;

    calib->width = intrin.width;
    calib->height = intrin.height;
    }

    rs_extrinsics extrin={0};
    rs_get_motion_extrinsics_from( device[devID].dev, device[devID].colorStreamToUse , &extrin, &e);

    if (!check_error())
    {
     fprintf(stderr,"Survived Extrinsics");

     calib->extrinsicParametersSet = 1;
     calib->extrinsic[0] = extrin.rotation[0]; calib->extrinsic[1] = extrin.rotation[1]; calib->extrinsic[2]  = extrin.rotation[2];   calib->extrinsic[3]=0;
     calib->extrinsic[4] = extrin.rotation[3]; calib->extrinsic[5] = extrin.rotation[4]; calib->extrinsic[6]  = extrin.rotation[5];   calib->extrinsic[7]=0;
     calib->extrinsic[8] = extrin.rotation[6]; calib->extrinsic[9] = extrin.rotation[7]; calib->extrinsic[10] = extrin.rotation[8];   calib->extrinsic[11]=0;
     calib->extrinsic[12]=0;                   calib->extrinsic[13]=0;                   calib->extrinsic[14]=0;                      calib->extrinsic[15]=1.0;

     calib->extrinsicTranslation[0] = extrin.translation[0];
     calib->extrinsicTranslation[1] = extrin.translation[1];
     calib->extrinsicTranslation[2] = extrin.translation[2];
    }


    return calib->intrinsicParametersSet;
}

int getRealsenseColorCalibration(int devID,struct calibration * calib)
{
    return getRealsenseGenericCalibration( device[devID].dev , device[devID].colorStreamToUse , calib );
}


int getRealsenseDepthCalibration(int devID,struct calibration * calib)
{
    return getRealsenseGenericCalibration( device[devID].dev , device[devID].depthStreamToUse , calib );
}
#endif // USE_CALIBRATION




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
