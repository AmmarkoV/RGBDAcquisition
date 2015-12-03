#include "opticalFlow.h"


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

using namespace cv;

int doLKOpticalFlow(cv::Mat leftRGB,cv::Mat lastLeftRGB)
{
    // winsize has to be 11 or 13, otherwise nothing is found
    int winsize = 11;
    int maxlvl = 5;

/*

    calcOpticalFlowPyrLK(imgAgray,imgBgray,cornersA,cornersB,status,error,Size(winsize, winsize), maxlvl);

    for (unsigned int i = 0; i < cornersB.size(); i++) {
        if (status[i] == 0 || error[i] > 0) {
            drawPixel(cornersB[i], &imgC, 2, red);
            continue;
        }
        drawPixel(cornersB[i], &imgC, 2, green);
        line(imgC, cornersA[i], cornersB[i], Scalar(255, 0, 0));
    }

    namedWindow("window", 1);
    moveWindow("window", 50, 50);

*/
}
