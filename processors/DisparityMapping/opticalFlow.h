#ifndef OPTICALFLOW_H_INCLUDED
#define OPTICALFLOW_H_INCLUDED



#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "cvaux.h"

int doLKOpticalFlow(cv::Mat leftRGB,cv::Mat lastLeftRGB);


#endif // OPTICALFLOW_H_INCLUDED
