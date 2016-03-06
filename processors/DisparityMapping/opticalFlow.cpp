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



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

using namespace cv;



vector<Point2f> previousCornersLeft;
vector<Point2f> nextCornersLeft;

vector<Point2f> previousCornersRight;
vector<Point2f> nextCornersRight;



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

int doLKOpticalFlow(cv::Mat leftBGR,cv::Mat previousImg,cv::Mat nextImg)
{

    // winsize has to be 11 or 13, otherwise nothing is found
    int winsize = 11;
    int maxlvl = 5;
    unsigned int maxFeaturesToUse=120;

    unsigned int sizeOfPixel=10;
    cv::Scalar red=cv::Scalar(255,0,0);
    cv::Scalar blue=cv::Scalar(0,0,255);
    cv::Scalar green=cv::Scalar(0,255,0);

    vector<Point2f> previousCorners;
    vector<Point2f> nextCorners;
    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10);




    goodFeaturesToTrack(previousImg, previousCorners, maxFeaturesToUse, 0.01, 30);
    cornerSubPix(previousImg, previousCorners , subPixWinSize, Size(-1,-1), termcrit);

    //goodFeaturesToTrack(previousImg, cornersB, 100, 0.01, 30);
    //cornerSubPix(previousImg, cornersB , subPixWinSize, Size(-1,-1), termcrit);
    //for (unsigned int i = 0; i < cornersB.size(); i++) { drawPixel(&flowVis , cornersB[i].x , cornersB[i].y , red , sizeOfPixel+5); }



    vector<uchar> status;
    vector<float> error;


    calcOpticalFlowPyrLK(previousImg,nextImg,previousCorners,nextCorners,status,error,Size(winsize, winsize), maxlvl);


    cv::Mat flowVis=leftBGR;
    for (unsigned int i = 0; i < previousCorners.size(); i++) { drawPixel(&flowVis , previousCorners[i].x , previousCorners[i].y , blue , sizeOfPixel); }
    for (unsigned int i = 0; i < nextCorners.size(); i++)
        {
          if (euclideanDist(previousCorners[i],nextCorners[i])>60)
              {
                drawPixel(&flowVis , nextCorners[i].x , nextCorners[i].y , red , sizeOfPixel);
              } else
          if (status[i] == 0 ) //|| error[i] > 0
              {
                drawPixel(&flowVis , nextCorners[i].x , nextCorners[i].y , red , sizeOfPixel);
              } else
              {
               drawPixel(&flowVis , nextCorners[i].x , nextCorners[i].y , green , sizeOfPixel);
               cv::line(flowVis, previousCorners[i], nextCorners[i],  green, 1, 8, 0);
              }
        }

    //imshow("window",flowVis);
    //namedWindow("window", 1);
    //moveWindow("window", 50, 50);

  return 1;
}











int doStereoLKOpticalFlow(
                           cv::Mat leftBGR ,cv::Mat nextLeftImg , cv::Mat previousLeftImg ,
                           cv::Mat rightBGR,cv::Mat nextRightImg ,cv::Mat previousRightImg
                         )
{
    // winsize has to be 11 or 13, otherwise nothing is found
    int winsize = 11;
    int maxlvl = 3;
    unsigned int maxFeaturesToUse=120;

    unsigned int sizeOfPixel=10;
    cv::Scalar red=cv::Scalar(255,0,0);
    cv::Scalar blue=cv::Scalar(0,0,255);
    cv::Scalar green=cv::Scalar(0,255,0);

    vector<uchar> status;
    vector<float> error;

    TermCriteria termcrit(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 20, 0.03);
    Size subPixWinSize(10,10);

    int forceRefreshOnEveryFrame = 1;

    previousCornersLeft = nextCornersLeft; nextCornersLeft.clear();
    fprintf(stderr,"Previous corners where %u  , Next corners are %u \n",previousCornersLeft.size(),nextCornersLeft.size());

    for (unsigned int i = 0; i < previousCornersLeft.size(); i++)
    {
      if
         ( (leftBGR.cols>previousCornersLeft[i].x ) ||
            (leftBGR.rows>previousCornersLeft[i].y )  )
      { fprintf(stderr,GREEN "ok " NORMAL); } else
      { fprintf(stderr,RED "err " NORMAL); }
      fprintf(stderr,"pt(%u,%0.2f,%0.2f)\n",i,previousCornersLeft[i].x,previousCornersLeft[i].y);
    }

    if ( (previousCornersLeft.size() < 50) || (forceRefreshOnEveryFrame) )
    {
     goodFeaturesToTrack(previousLeftImg, previousCornersLeft, maxFeaturesToUse, 0.01, 30);
     cornerSubPix(previousLeftImg, previousCornersLeft , subPixWinSize, Size(-1,-1), termcrit);
    }


    previousCornersRight = nextCornersRight; nextCornersRight.clear();
    fprintf(stderr,"Previous corners where %u , Next corners are %u  \n",previousCornersRight.size(),nextCornersRight.size());
    if ( (previousCornersRight.size() < 50) || (forceRefreshOnEveryFrame) )
    {
     goodFeaturesToTrack(previousRightImg, previousCornersRight, maxFeaturesToUse, 0.01, 30);
     cornerSubPix(previousRightImg, previousCornersRight , subPixWinSize, Size(-1,-1), termcrit);
    }

    nextCornersLeft.clear();
    nextCornersRight.clear();

    calcOpticalFlowPyrLK(previousLeftImg,nextLeftImg,previousCornersLeft,nextCornersLeft,status,error,Size(winsize, winsize), maxlvl);
    calcOpticalFlowPyrLK(previousRightImg,nextRightImg,previousCornersRight,nextCornersRight,status,error,Size(winsize, winsize), maxlvl);


//Do Drawing
    vector<Point2f> ptA=previousCornersLeft;
    vector<Point2f> ptB=nextCornersLeft;
    cv::Mat flowVis=leftBGR;
    int i=0;
    for (i=0; i<2; i++)
    {
     if (i==0) { ptA=previousCornersLeft;  ptB=nextCornersLeft;  flowVis=leftBGR;  } else
     if (i==1) { ptA=previousCornersRight; ptB=nextCornersRight; flowVis=rightBGR; }

     for (unsigned int i = 0; i < ptA.size(); i++) { drawPixel(&flowVis , ptA[i].x , ptA[i].y , blue , sizeOfPixel); }
     for (unsigned int i = 0; i < ptB.size(); i++)
        {
          drawPixel(&flowVis , ptB[i].x , ptB[i].y , blue , sizeOfPixel);
          if (euclideanDist(ptA[i],ptB[i])>60)
              {
                drawPixel(&flowVis , ptB[i].x , ptB[i].y , red , sizeOfPixel);
              } else
          if (status[i] == 0 ) //|| error[i] > 0
              {
                drawPixel(&flowVis , ptB[i].x , ptB[i].y , red , sizeOfPixel);
              } else
              {
               drawPixel(&flowVis , ptB[i].x , ptB[i].y , green , sizeOfPixel);
               cv::line(flowVis, ptA[i], ptB[i],  green, 1, 8, 0);
              }
        }

   }


}



