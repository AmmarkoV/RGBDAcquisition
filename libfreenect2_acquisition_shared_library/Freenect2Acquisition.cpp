
#include <stdio.h>
#include <stdlib.h>

#include "Freenect2Acquisition.h"

#if BUILD_FREENECT2
#include "../3dparty/libfreenect/include/libfreenect.h"
#include "../3dparty/libfreenect/wrappers/c_sync/libfreenect_sync.h"

#define MAX_DEVS 16

int rgb_mode[MAX_DEVS]={FREENECT_VIDEO_RGB};
int depth_mode[MAX_DEVS]={FREENECT_DEPTH_11BIT};


int convertAndSave(unsigned int framesRecorded , libfreenect2::Frame *  rgb ,  libfreenect2::Frame *  ir , libfreenect2::Frame * depth)
{
    //--------------------------------------------------------------------------
    cv::Mat rgbMat(rgb->height, rgb->width, CV_8UC3, rgb->data);

    cv::Mat depthMat(depth->height, depth->width, CV_32FC1, depth->data) ;
    //cv::Mat depthMatR = depthMat / 4500.0f;
    //--------------------------------------------------------------------------


    #if USE_REAL_COLOR
     unsigned int dW = depth->width-1;
     unsigned int dH = (rgb->height / 3 ) -1;

     cv::Mat rgbMatAligned(rgb->height / 3 , rgb->width / 3 , CV_8UC3);
     cv::resize(rgbMat, rgbMatAligned, rgbMatAligned.size(), 0.5, 0.5, cv::INTER_LINEAR );

    cv::Mat rgbMatAlignedOk(dH, dW , CV_8UC3);
    //---------
    //where image starts

    cv::Rect roi( (rgb->width/3 - depth->width)/2 , 0 ,    dW , dH  );
    cv::Mat image_roi = rgbMatAligned(roi);
    image_roi.copyTo(rgbMatAlignedOk);


    //cv::imshow("rgbAligned",rgbMatAligned);
    //cv::imshow("rgbAlignedOk",rgbMatAlignedOk);
    #else
    cv::Mat irMat(ir->height, ir->width, CV_32FC1, ir->data);
    cv::Mat irMatR = irMat / 20000.0f;

    cv::Mat irMatUC(ir->height, ir->width, CV_8UC1);
    irMatR.convertTo(irMatUC, CV_8UC1, 254, 0);
    cv::Mat irRGBMat(ir->height, ir->width, CV_8UC3);

    cvtColor(irMatUC, irRGBMat, CV_GRAY2RGB);
    cv::imshow("ir", irRGBMat);
    #endif // USE_REAL_COLOR


    cv::Mat depthMatUS(depth->height, depth->width,  CV_32FC1) ;
    //depthMatR.convertTo(depthMatUS, CV_16UC1, 4500, 0);
    depthMat.convertTo(depthMatUS, CV_16UC1);
    //cv::imshow("depth", depthMatR);

    char filename[FILENAME_MAX]={0};
    // Write File down ---------------------------------------------------


    #if USE_REAL_COLOR
    snprintf(filename,FILENAME_MAX,"frames/kinect2/colorFrameBig_0_%05u.png",framesRecorded);
    cv::imwrite(filename,rgbMatAligned);

    snprintf(filename,FILENAME_MAX,"frames/kinect2/colorFrame_0_%05u.png",framesRecorded);
    cv::imwrite(filename,rgbMatAlignedOk);
    #else
     snprintf(filename,FILENAME_MAX,"frames/kinect2/colorFrame_0_%05u.png",framesRecorded);
     cv::imwrite(filename,irRGBMat);
    #endif



    snprintf(filename,FILENAME_MAX,"frames/kinect2/depthFrame_0_%05u.png",framesRecorded);
    cv::imwrite(filename,depthMatUS);
    // -------------------------------------------------------------------

  return 1;
}


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
