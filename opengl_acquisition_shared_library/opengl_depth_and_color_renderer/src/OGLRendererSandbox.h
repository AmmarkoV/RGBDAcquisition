#ifndef OGLRENDERERSANDBOX_H_INCLUDED
#define OGLRENDERERSANDBOX_H_INCLUDED

#ifdef __cplusplus
extern "C"
{
#endif

int getOpenGLDepth(short * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height);
int getOpenGLColor(char * depth , unsigned int x,unsigned int y,unsigned int width,unsigned int height);

double getOpenGLFocalLength();
double getOpenGLPixelSize();


int startOGLRendererSandbox();
int snapOGLRendererSandbox();
int stopOGLRendererSandbox();

#ifdef __cplusplus
}
#endif

#endif // OGLRENDERERSANDBOX_H_INCLUDED
