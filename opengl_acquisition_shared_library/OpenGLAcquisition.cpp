#include "OpenGLAcquisition.h"
#include "../acquisition/Acquisition.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opengl_depth_and_color_renderer/src/OGLRendererSandbox.h"

unsigned int openGL_WIDTH=640;
unsigned int openGL_HEIGHT=480;
char * openGLColorFrame = 0;
short * openGLDepthFrame = 0;

struct calibration calibRGB;
struct calibration calibDepth;


int startOpenGL(unsigned int max_devs)
{
  return 1;
}

int getOpenGLNumberOfDevices()
{
   return 1;
}

int stopOpenGL()
{
   return 1;
}

int createOpenGLDevice(int devID,unsigned int width,unsigned int height,unsigned int framerate)
{
  if ( ( openGL_WIDTH < width ) &&  ( openGL_HEIGHT < height ) )
   {
        openGL_HEIGHT=height;
        openGL_WIDTH=width;
   }

  if(openGLColorFrame!=0) { openGLColorFrame= (char*) realloc(openGLColorFrame,sizeof(char) * openGL_WIDTH*openGL_HEIGHT*3); } else
                          { openGLColorFrame = (char*)  malloc(sizeof(char) * openGL_WIDTH*openGL_HEIGHT*3); }

  if(openGLDepthFrame!=0) { openGLDepthFrame= (short*) realloc(openGLDepthFrame,sizeof(short) * openGL_WIDTH*openGL_HEIGHT*1); } else
                          { openGLDepthFrame = (short*)  malloc(sizeof(short) * openGL_WIDTH*openGL_HEIGHT*1); }

  startOGLRendererSandbox();

  return ((openGLColorFrame!=0) && (openGLDepthFrame!=0)) ;
}



int destroyOpenGLDevice(int devID)
{
  stopOGLRendererSandbox();
  if (openGLColorFrame!=0) { free(openGLColorFrame); openGLColorFrame=0; }
  if (openGLDepthFrame!=0) { free(openGLDepthFrame); openGLDepthFrame=0; }
  return 1;
}


int snapOpenGLFrames(int devID) { return snapOGLRendererSandbox(); }

//Color Frame getters
int getOpenGLColorWidth(int devID) { return openGL_WIDTH; }
int getOpenGLColorHeight(int devID) { return openGL_HEIGHT; }
int getOpenGLColorDataSize(int devID) { return openGL_HEIGHT*openGL_WIDTH * 3; }
int getOpenGLColorChannels(int devID)     { return 3; }
int getOpenGLColorBitsPerPixel(int devID) { return 8; }
char * getOpenGLColorPixels(int devID)
{
  getOpenGLColor(openGLColorFrame,0,0,getOpenGLColorWidth(devID),getOpenGLColorHeight(devID));
 return openGLColorFrame;
}

double getOpenGLColorFocalLength(int devID)
{
   fprintf(stderr,"getOpenGLDepthFocalLength returns fixed values.. \n");
   //return 120.0;
   return getOpenGLFocalLength();
}

double getOpenGLColorPixelSize(int devID)
{
    fprintf(stderr,"getOpenGLDepthPixelSize returns fixed values.. \n");
    //return 0.1052;
    return getOpenGLPixelSize();
}


int getOpenGLColorCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &calibRGB,sizeof(struct calibration));
    return 1;
}

int getOpenGLDepthCalibration(int devID,struct calibration * calib)
{
    memcpy((void*) calib,(void*) &calibDepth,sizeof(struct calibration));
    return 1;
}

int setOpenGLColorCalibration(int devID,struct calibration * calib)
{
    setOpenGLExtrinsicCalibration((float*) calib->extrinsicRotationRodriguez,(float*) calib->extrinsicTranslation);
    return 1;
}

int setOpenGLDepthCalibration(int devID,struct calibration * calib)
{
    setOpenGLExtrinsicCalibration((float*) calib->extrinsicRotationRodriguez,(float*) calib->extrinsicTranslation);
    return 1;
}



   //Depth Frame getters
int getOpenGLDepthWidth(int devID)    {  return openGL_WIDTH; }
int getOpenGLDepthHeight(int devID)   { return openGL_HEIGHT; }
int getOpenGLDepthDataSize(int devID) { return openGL_WIDTH*openGL_HEIGHT; }
int getOpenGLDepthChannels(int devID)     { return 1; }
int getOpenGLDepthBitsPerPixel(int devID) { return 16; }

char * getOpenGLDepthPixels(int devID)
{
  getOpenGLDepth(openGLDepthFrame,0,0,getOpenGLDepthWidth(devID),getOpenGLDepthHeight(devID));
  return (char*) openGLDepthFrame;
}

double getOpenGLDepthFocalLength(int devID)
{
   fprintf(stderr,"getOpenGLDepthFocalLength returns fixed values.. \n");
   //return 120.0;
   return getOpenGLFocalLength();
}

double getOpenGLDepthPixelSize(int devID)
{
    fprintf(stderr,"getOpenGLDepthPixelSize returns fixed values.. \n");
    //return 0.1052;
    return getOpenGLPixelSize();
}

