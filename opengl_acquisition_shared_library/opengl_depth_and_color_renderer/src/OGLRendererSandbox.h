#ifndef OGLRENDERERSANDBOX_H_INCLUDED
#define OGLRENDERERSANDBOX_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

int setOpenGLNearFarPlanes(double near , double far);
int setOpenGLIntrinsicCalibration(double * camera);
int setOpenGLExtrinsicCalibration(double * rodriguez,double * translation);

int getOpenGLDepth(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height);
int getOpenGLColor(char * color , unsigned int x,unsigned int y,unsigned int width,unsigned int height);

void writeOpenGLColor(char * colorfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height);
void writeOpenGLDepth(char * depthfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height);

double getOpenGLFocalLength();
double getOpenGLPixelSize();


int startOGLRendererSandbox(char * sceneFile);
int snapOGLRendererSandbox();
int stopOGLRendererSandbox();

#ifdef __cplusplus
}
#endif

#endif // OGLRENDERERSANDBOX_H_INCLUDED
