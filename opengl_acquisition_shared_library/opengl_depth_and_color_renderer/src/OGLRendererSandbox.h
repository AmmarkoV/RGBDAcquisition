#ifndef OGLRENDERERSANDBOX_H_INCLUDED
#define OGLRENDERERSANDBOX_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

int setOpenGLNearFarPlanes(double near , double far);
int setOpenGLIntrinsicCalibration(double * camera);
int setOpenGLExtrinsicCalibration(double * rodriguez,double * translation);


unsigned int getOpenGLWidth();
unsigned int getOpenGLHeight();

int getOpenGLDepth(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height);
int getOpenGLColor(char * color , unsigned int x,unsigned int y,unsigned int width,unsigned int height);

void writeOpenGLColor(char * colorfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height);
void writeOpenGLDepth(char * depthfile,unsigned int x,unsigned int y,unsigned int width,unsigned int height);

double getOpenGLFocalLength();
double getOpenGLPixelSize();


int startOGLRendererSandbox(unsigned int width,unsigned int height , unsigned int viewWindow ,char * sceneFile);
int snapOGLRendererSandbox();
int snapOGLRendererPhotoshootSandbox(int objID, unsigned int columns , unsigned int rows , float distance,
                                     float angleX,float angleY,float angleZ,
                                     float angXVariance ,float angYVariance , float angZVariance
                                    );
int stopOGLRendererSandbox();

#ifdef __cplusplus
}
#endif

#endif // OGLRENDERERSANDBOX_H_INCLUDED
