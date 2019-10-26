#include <stdio.h>
#include <stdlib.h>

#include "OpenGLAcquisition.h"
#define USE_SIMPLE_FAST_DEPTH 0

#if BUILD_OPENGL

#include "../acquisition/Acquisition.h"
#include <string.h>

#include "opengl_depth_and_color_renderer/src/Library/OGLRendererSandbox.h"

unsigned int openGL_WIDTH=640;
unsigned int openGL_HEIGHT=480;
unsigned int openGL_Framerate=0;
char * openGLColorFrame = 0;
short * openGLDepthFrame = 0;

struct calibration calibRGB;
struct calibration calibDepth;

int tryStartingWithoutWindow=1;


int startOpenGLModule(unsigned int max_devs,const char * settings)
{
    if (strstr(settings,"safe")!=0)
    {
        tryStartingWithoutWindow=0;
    }
  return 1;
}

int getOpenGLNumberOfDevices()
{
   return 1;
}

int stopOpenGLModule()
{
   return 1;
}

int createOpenGLDevice(int devID,const char * devName,unsigned int width,unsigned int height,unsigned int framerate)
{
  if ( ( openGL_WIDTH <= width ) &&  ( openGL_HEIGHT <= height ) )
   {
        openGL_HEIGHT=height;
        openGL_WIDTH=width;
   }

   openGL_Framerate=framerate;

  if(openGLColorFrame!=0) { openGLColorFrame= (char*) realloc(openGLColorFrame,sizeof(char) * openGL_WIDTH*openGL_HEIGHT*3); } else
                          { openGLColorFrame = (char*)  malloc(sizeof(char) * openGL_WIDTH*openGL_HEIGHT*3); }

  if(openGLDepthFrame!=0) { openGLDepthFrame= (short*) realloc(openGLDepthFrame,sizeof(short) * openGL_WIDTH*openGL_HEIGHT*1); } else
                          { openGLDepthFrame = (short*)  malloc(sizeof(short) * openGL_WIDTH*openGL_HEIGHT*1); }


    if (!tryStartingWithoutWindow)
    {
       if (!startOGLRendererSandbox(0,0,openGL_WIDTH,openGL_HEIGHT,1 /*View Window*/,devName) )
      {
        return 0;
        //return ((openGLColorFrame!=0) && (openGLDepthFrame!=0)) ;
      }
    }
    else
   if  (!startOGLRendererSandbox(0,0,openGL_WIDTH,openGL_HEIGHT,0 /*View Window*/,devName) )
   {
     fprintf(stderr,"Could not start openGL context with a p-buffer..");
     fprintf(stderr,"Will now try to start it with a visible window..");
     if (!startOGLRendererSandbox(0,0,openGL_WIDTH,openGL_HEIGHT,1 /*View Window*/,devName) )
     {
        return 0;
        //return ((openGLColorFrame!=0) && (openGLDepthFrame!=0)) ;
     }
   }


   //THis is scene controlled , but one could force the grabber to render frame by frame uncommenting the following
   //Switch OGL Renderer to sequential mode so each frame snap will provide the next available rendering and nothing will be lost..!
   //changeOGLRendererGrabMode(1);

  return ((openGLColorFrame!=0) && (openGLDepthFrame!=0)) ;
}


int controlOpenGLScene(const char * name,const char * variable,int control,float value)
{
 fprintf(stderr,"Object %s -> variable %s[%u] is set to %0.2f \n",name,variable,control,value);
 return controlScene(name,variable,control,value);
}



int passUserCommandOpenGL(const char * command,const char * value)
{
 return passUserCommand(command,value);
}

int passUserInputOpenGL(int devID,char key,int state,unsigned int x, unsigned int y)
{
 return passUserInput(key,state,x,y);
}

int destroyOpenGLDevice(int devID)
{
  stopOGLRendererSandbox();
  if (openGLColorFrame!=0) { free(openGLColorFrame); openGLColorFrame=0; }
  if (openGLDepthFrame!=0) { free(openGLDepthFrame); openGLDepthFrame=0; }
  return 1;
}


int seekRelativeOpenGLFrame(int devID,signed int seekFrame) { return seekRelativeOGLRendererSandbox(devID,seekFrame); }

int seekOpenGLFrame(int devID,unsigned int seekFrame) {  return seekOGLRendererSandbox(devID,seekFrame); }

int snapOpenGLFrames(int devID) { return snapOGLRendererSandbox(openGL_Framerate); }

//Color Frame getters
unsigned long getLastOpenGLColorTimestamp(int devID) { return getOpenGLTimestamp(); }
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

int setOpenGLCalibration(int devID,struct calibration * calib)
{
    fprintf(stderr,"setOpenGLCalibration(0,calib) called \n");

    double * rodriguez = (double*) malloc(sizeof(double) * 3 );
    double * translation = (double*) malloc(sizeof(double) * 3 );
    double * camera = (double*) malloc(sizeof(double) * 9 );

    int i=0;
    for (i=0; i<3; i++) { rodriguez[i]=calib->extrinsicRotationRodriguez[i]; }
    for (i=0; i<3; i++) { translation[i]=calib->extrinsicTranslation[i]; }
    for (i=0; i<9; i++) { camera[i]=calib->intrinsic[i]; }

    fprintf(stderr,"Setting OpenGL near(%0.2f)/far(%0.2f) planes\n",calib->nearPlane , calib->farPlane);
    setOpenGLNearFarPlanes( calib->nearPlane , calib->farPlane );
    fprintf(stderr,"Setting Intrinsics for OpenGL\n");
    setOpenGLIntrinsicCalibration(camera);


    if ( (rodriguez[0]!=0.0) || (rodriguez[1]!=0.0) || (rodriguez[2]!=0.0) ||
         (translation[0]!=0.0) || (translation[1]!=0.0) || (translation[2]!=0.0)   )
      {
        fprintf(stderr,"Setting Extrinsics for OpenGL\n");
        setOpenGLExtrinsicCalibration(rodriguez,translation,calib->depthUnit);
      } else
      {
        fprintf(stderr,"NOT setting Extrinsics for OpenGL\n");
      }

    free(rodriguez);
    free(translation);
    free(camera);
    return 1;
}


int setOpenGLColorCalibration(int devID,struct calibration * calib)
{
    fprintf(stderr,"setOpenGLColorCalibration(0,calib) called \n");
    return setOpenGLCalibration(devID,calib);
}

int setOpenGLDepthCalibration(int devID,struct calibration * calib)
{
    fprintf(stderr,"setOpenGLDepthCalibration(0,calib) called \n");
    return setOpenGLCalibration(devID,calib);
}



   //Depth Frame getters
unsigned long getLastOpenGLDepthTimestamp(int devID) { return getOpenGLTimestamp(); }
int getOpenGLDepthWidth(int devID)    {  return openGL_WIDTH; }
int getOpenGLDepthHeight(int devID)   { return openGL_HEIGHT; }
int getOpenGLDepthDataSize(int devID) { return openGL_WIDTH*openGL_HEIGHT; }
int getOpenGLDepthChannels(int devID)     { return 1; }
int getOpenGLDepthBitsPerPixel(int devID) { return 16; }

char * getOpenGLDepthPixels(int devID)
{
  #if USE_SIMPLE_FAST_DEPTH
    getOpenGLZBuffer(openGLDepthFrame,0,0,getOpenGLDepthWidth(devID),getOpenGLDepthHeight(devID));
  #else
   //Real depth gets returned using the next function , problem is that
   //most times we dont want real depth..
   getOpenGLDepth(openGLDepthFrame,0,0,getOpenGLDepthWidth(devID),getOpenGLDepthHeight(devID));
  #endif // USE_SIMPLE_FAST_DEPTH


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

#else
//Null build
int startOpenGLModule(unsigned int max_devs,char * settings)
{
    fprintf(stderr,"startOpenGLModule called on a dummy build of OpenGLAcquisition!\n");
    fprintf(stderr,"Please consider enabling #define BUILD_OPENGL 1 on acquisition/acquisition_setup.h\n");
    return 0;
  return 1;
}
#endif
