#ifndef IMAGESTREAM_H_INCLUDED
#define IMAGESTREAM_H_INCLUDED

#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "cvaux.h"

extern cv::Mat leftRGB,lastLeftRGB;
extern cv::Mat rightRGB,lastRightRGB;

extern cv::Mat greyLeft, greyLastLeft;
extern cv::Mat greyRight, greyLastRight;

int passNewFrame(unsigned char * colorFrame , unsigned int colorWidth ,unsigned int colorHeight , unsigned int swapColorFeeds , unsigned int shiftYLeft , unsigned int shiftYRight );

#endif // IMAGESTREAM_H_INCLUDED
