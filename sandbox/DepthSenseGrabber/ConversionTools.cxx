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



FrameFormat formatName(int resType) {
    switch (resType) {
        case 0: return FRAME_FORMAT_QQVGA;
        case 1: return FRAME_FORMAT_QVGA;
        case 2: return FRAME_FORMAT_VGA;
        case 3: return FRAME_FORMAT_WXGA_H;
        case 4: return FRAME_FORMAT_NHD;
        default: printf("Unsupported resolution parameter\n"); return FRAME_FORMAT_WXGA_H;
    }
}
int formatResX(int resType) {
    switch (resType) {
        case 0: return 160;
        case 1: return 320;
        case 2: return 640;
        case 3: return 1280;
        case 4: return 640;
        default: printf("Unsupported resolution parameter\n"); return 0;
    }
}
int formatResY(int resType) {
    switch (resType) {
        case 0: return 120;
        case 1: return 240;
        case 2: return 480;
        case 3: return 720;
        case 4: return 360;
        default: printf("Unsupported resolution parameter\n"); return 0;
    }
}



unsigned char * convertShortDepthToCharDepth(unsigned short * depth,unsigned int width , unsigned int height , unsigned int min_depth , unsigned int max_depth)
{
  if (depth==0)  { fprintf(stderr,"Depth is not allocated , cannot perform DepthToRGB transformation \n"); return 0; }
  unsigned short * depthPTR= depth; // This will be the traversing pointer for input
  unsigned short * depthLimit =  depth + width*height; //<- we use sizeof(short) because we have casted to char !


  unsigned char * outFrame = (unsigned char*) malloc(width*height*1*sizeof(char));
  if (outFrame==0) { fprintf(stderr,"Could not perform DepthToRGB transformation\nNo memory for new frame\n"); return 0; }

  float depth_range = max_depth-min_depth;
  if (depth_range ==0 ) { depth_range = 1; }
  float multiplier = 255 / depth_range;


  unsigned char * outFramePTR = outFrame; // This will be the traversing pointer for output
  while ( depthPTR<depthLimit )
  {
     unsigned int scaled = (unsigned int) (*depthPTR) * multiplier;
     unsigned char scaledChar = (unsigned char) scaled;
     * outFramePTR = scaledChar;

     ++outFramePTR;
     ++depthPTR;
  }
 return outFrame;
}




// From SoftKinetic
// convert a YUY2 image to RGB


void yuy2rgb(unsigned char *dst, const unsigned char *src, const int width, const int height) {
  int x, y;
  const int width2 = width * 2;
  const int width4 = width * 3;
  const unsigned char *src1 = src;
  unsigned char *dst1 = dst;

  for (y=0; y<height; y++) {
    for (x=0; x<width; x+=2) {
      int x2=x*2;
      int y1  = src1[x2  ];
      int y2  = src1[x2+2];
      int u   = src1[x2+1] - 128;
      int v   = src1[x2+3] - 128;
      int uvr = (          15748 * v) / 10000;
      int uvg = (-1873 * u - 4681 * v) / 10000;
      int uvb = (18556 * u          ) / 10000;

      int r1 = y1 + uvr;
      int r2 = y2 + uvr;
      int g1 = y1 + uvg;
      int g2 = y2 + uvg;
      int b1 = y1 + uvb;
      int b2 = y2 + uvb;

      int x4=x*3;
      dst1[x4+0] = (b1 > 255) ? 255 : ((b1 < 0) ? 0 : b1);
      dst1[x4+1] = (g1 > 255) ? 255 : ((g1 < 0) ? 0 : g1);
      dst1[x4+2] = (r1 > 255) ? 255 : ((r1 < 0) ? 0 : r1);
      //dst1[x4+3] = 255;

      dst1[x4+3] = (b2 > 255) ? 255 : ((b2 < 0) ? 0 : b2);
      dst1[x4+4] = (g2 > 255) ? 255 : ((g2 < 0) ? 0 : g2);
      dst1[x4+5] = (r2 > 255) ? 255 : ((r2 < 0) ? 0 : r2);
    }
    src1 += width2;
    dst1 += width4;
  }
}
