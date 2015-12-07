#ifndef OPTICALFLOW_H_INCLUDED
#define OPTICALFLOW_H_INCLUDED



#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "cvaux.h"

int doLKOpticalFlow(cv::Mat leftBGR,cv::Mat Gray,cv::Mat lastGray);


#endif // OPTICALFLOW_H_INCLUDED
