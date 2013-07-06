#ifndef OPENGL_ACQUISITION_H_INCLUDED
#define OPENGL_ACQUISITION_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif


int startOpenGL(unsigned int max_devs) ;
int getOpenGLNumberOfDevices() ;
int stopOpenGL() ;
int createOpenGLDevice(int devID,unsigned int width,unsigned int height,unsigned int framerate) ;
int destroyOpenGLDevice(int devID) ;
int snapOpenGLFrames(int devID) ;

//Color Frame getters
int getOpenGLColorWidth(int devID) ;
int getOpenGLColorHeight(int devID) ;
int getOpenGLColorDataSize(int devID);
int getOpenGLColorChannels(int devID);
int getOpenGLColorBitsPerPixel(int devID);
char * getOpenGLColorPixels(int devID) ;
int saveOpenGLColorFramePNM(int devID,char * filename) ;

double getOpenGLColorFocalLength(int devID);
double getOpenGLColorPixelSize(int devID);


int getOpenGLColorCalibration(int devID,struct calibration * calib);
int getOpenGLDepthCalibration(int devID,struct calibration * calib);

int setOpenGLColorCalibration(int devID,struct calibration * calib);
int setOpenGLDepthCalibration(int devID,struct calibration * calib);

   //Depth Frame getters
int getOpenGLDepthWidth(int devID) ;
int getOpenGLDepthHeight(int devID) ;
int getOpenGLDepthDataSize(int devID) ;
int getOpenGLDepthChannels(int devID);
int getOpenGLDepthBitsPerPixel(int devID);
char * getOpenGLDepthPixels(int devID) ;
int saveOpenGLDepthFramePNM(int devID,char * filename) ;

double getOpenGLDepthFocalLength(int devID);
double getOpenGLDepthPixelSize(int devID);


#ifdef __cplusplus
}
#endif

#endif // TEMPLATEACQUISITION_H_INCLUDED
