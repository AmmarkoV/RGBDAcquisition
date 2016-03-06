#ifndef OPTICALFLOW_H_INCLUDED
#define OPTICALFLOW_H_INCLUDED



#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "cvaux.h"

int doLKOpticalFlow(cv::Mat leftBGR,cv::Mat previousImg,cv::Mat nextImg);

int doStereoLKOpticalFlow(
                           cv::Mat leftBGR,cv::Mat nextLeftImg ,cv::Mat previousLeftImg ,
                           cv::Mat rightBGR,cv::Mat nextRightImg ,cv::Mat previousRightImg
                         );


#endif // OPTICALFLOW_H_INCLUDED
