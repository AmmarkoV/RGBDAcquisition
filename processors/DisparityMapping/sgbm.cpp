#include "sgbm.h"


#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "cvaux.h"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

#include "stereo_calibrate.h"



int doSGBM(unsigned char * colorFrame , unsigned int colorWidth ,unsigned int colorHeight )
{
    cv::Mat rgbImg(colorHeight,colorWidth,CV_8UC3,colorFrame);
    //cv::Mat depthImg(depthHeight,depthWidth,CV_16UC1,depthFrame);


    cv::imshow("test",rgbImg);
    cv::waitKey(5);


}
