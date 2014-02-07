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

void saveRawColorFrame(char* fileName, uint8_t* pixels, int width, int height, int timeStamp);
void saveRawDepthFrame(char* fileName, unsigned short* pixels, int width, int height, int timeStamp);
