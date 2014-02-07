#ifdef _MSC_VER
#include <windows.h>
#endif

#include <stdio.h>
#include <time.h>

#include <vector>
#include <exception>

#include "cv.h"
#include "highgui.h"

#include <DepthSense.hxx>

using namespace DepthSense;
using namespace std;


FrameFormat formatName(int resType);
int formatResX(int resType);
int formatResY(int resType);

unsigned char * convertShortDepthToCharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth);

// From SoftKinetic
// convert a YUY2 image to RGB
void yuy2rgb(unsigned char *dst, const unsigned char *src, const int width, const int height);
