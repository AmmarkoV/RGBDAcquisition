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


int doLKOpticalFlow(cv::Mat leftBGR,cv::Mat leftGray,cv::Mat lastLeftGray)
{
    cv::Mat flowVis=leftBGR;

    // winsize has to be 11 or 13, otherwise nothing is found
    int winsize = 11;
    int maxlvl = 5;

    unsigned int sizeOfPixel=10;
    cv::Scalar blue=cv::Scalar(255,0,0);
    cv::Scalar red=cv::Scalar(0,0,255);
    cv::Scalar green=cv::Scalar(0,255,0);

    vector<Point2f> cornersA;
    vector<Point2f> cornersB;

    goodFeaturesToTrack(leftGray, cornersA, 100, 0.01, 30);
    for (unsigned int i = 0; i < cornersA.size(); i++) { drawPixel(&flowVis , cornersA[i].x , cornersA[i].y , blue , sizeOfPixel); }



/*
    Point textPos=Point(100,100);
    char outBuf[1024]={0};
    snprintf(outBuf,1024,"this %u , last %u \n",cornersA.size() , cornersB.size() );
    putText( flowVis,outBuf, textPos , CV_FONT_HERSHEY_COMPLEX, 3,  green , 5, 2 );
*/

    // I have no idea what does it do
//    cornerSubPix(imgAgray, cornersA, Size(15, 15), Size(-1, -1),
//            TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 20, 0.03));

    vector<uchar> status;
    vector<float> error;


    calcOpticalFlowPyrLK(lastLeftGray,leftGray,cornersA,cornersB,status,error,Size(winsize, winsize), maxlvl);

    for (unsigned int i = 0; i < cornersB.size(); i++)
        {
         if (status[i] == 0 || error[i] > 0) { drawPixel(&flowVis , cornersB[i].x , cornersB[i].y , red , sizeOfPixel);   continue; }
         drawPixel(&flowVis , cornersB[i].x , cornersB[i].y , green , sizeOfPixel);
         line(flowVis, cornersA[i], cornersB[i], Scalar(255, 0, 0));
        }

    //imshow("window",flowVis);
    //namedWindow("window", 1);
    //moveWindow("window", 50, 50);

  return 1;
}
