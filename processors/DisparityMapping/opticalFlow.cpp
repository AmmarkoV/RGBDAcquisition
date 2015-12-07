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

void drawPixel(cv::Mat * img , unsigned int x, unsigned int y , cv::Scalar color,unsigned int sizeOfPixel)
{
    Point pt1=Point(x-sizeOfPixel,y-sizeOfPixel);
    Point pt2=Point(x+sizeOfPixel,y+sizeOfPixel);
    rectangle(*img,pt1,pt2,color,1,8,0);
}


float euclideanDist(cv::Point2f& p, cv::Point2f& q)
{
    cv::Point diff = p - q;
    return cv::sqrt(diff.x*diff.x + diff.y*diff.y);
}

int doLKOpticalFlow(cv::Mat leftBGR,cv::Mat Gray,cv::Mat lastGray)
{
    cv::Mat flowVis=leftBGR;

    // winsize has to be 11 or 13, otherwise nothing is found
    int winsize = 11;
    int maxlvl = 5;
    unsigned int maxFeaturesToUse=120;

    unsigned int sizeOfPixel=10;
    cv::Scalar red=cv::Scalar(255,0,0);
    cv::Scalar blue=cv::Scalar(0,0,255);
    cv::Scalar green=cv::Scalar(0,255,0);

    vector<Point2f> cornersA;
    vector<Point2f> cornersB;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10);

    goodFeaturesToTrack(Gray, cornersA, maxFeaturesToUse, 0.01, 30);
    cornerSubPix(Gray, cornersA , subPixWinSize, Size(-1,-1), termcrit);
    for (unsigned int i = 0; i < cornersA.size(); i++) { drawPixel(&flowVis , cornersA[i].x , cornersA[i].y , blue , sizeOfPixel); }

    //goodFeaturesToTrack(lastGray, cornersB, 100, 0.01, 30);
    //cornerSubPix(lastGray, cornersB , subPixWinSize, Size(-1,-1), termcrit);
    //for (unsigned int i = 0; i < cornersB.size(); i++) { drawPixel(&flowVis , cornersB[i].x , cornersB[i].y , red , sizeOfPixel+5); }



    vector<uchar> status;
    vector<float> error;


    calcOpticalFlowPyrLK(lastGray,Gray,cornersA,cornersB,status,error,Size(winsize, winsize), maxlvl);
    for (unsigned int i = 0; i < cornersB.size(); i++)
        {
          if (euclideanDist(cornersA[i],cornersB[i])>60)
              {
                drawPixel(&flowVis , cornersB[i].x , cornersB[i].y , red , sizeOfPixel);
              } else
          if (status[i] == 0 ) //|| error[i] > 0
              {
                drawPixel(&flowVis , cornersB[i].x , cornersB[i].y , red , sizeOfPixel);
              } else
              {
               drawPixel(&flowVis , cornersB[i].x , cornersB[i].y , green , sizeOfPixel);
               cv::line(flowVis, cornersA[i], cornersB[i],  green, 1, 8, 0);
              }
        }

    //imshow("window",flowVis);
    //namedWindow("window", 1);
    //moveWindow("window", 50, 50);

  return 1;
}
