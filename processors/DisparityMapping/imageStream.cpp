#include "imageStream.h"

cv::Mat leftRGB,lastLeftRGB;
cv::Mat rightRGB,lastRightRGB;

cv::Mat greyLeft, greyLastLeft;
cv::Mat greyRight, greyLastRight;


unsigned int framesProcessed=0;


int passNewFrame(unsigned char * colorFrame , unsigned int colorWidth ,unsigned int colorHeight , unsigned int swapColorFeeds , unsigned int shiftYLeft , unsigned int shiftYRight )
{
    cv::Mat rgbImg(colorHeight,colorWidth,CV_8UC3,colorFrame);
    //cv::Mat depthImg(depthHeight,depthWidth,CV_16UC1,depthFrame);

    std::cerr<<"Doing Left split of images..\n";

    cv::Mat tmpleftImage;
    cv::Mat tmprightImage;
    cv::Rect leftROI;
    cv::Rect rightROI;


    leftROI = cv::Rect(0,0,(rgbImg.cols/2)-1,rgbImg.rows-1);
    rightROI = cv::Rect(rgbImg.cols/2,0,(rgbImg.cols/2)-1,rgbImg.rows-1);

    if (swapColorFeeds)
    {
     tmpleftImage= cv::Mat(rgbImg,rightROI);
     tmprightImage= cv::Mat(rgbImg,leftROI);
    }else
    {
     tmpleftImage= cv::Mat(rgbImg,leftROI);
     tmprightImage= cv::Mat(rgbImg,rightROI);
    }

    cv::Mat bgrleftImage = cv::Mat::zeros(tmpleftImage.size(), tmpleftImage.type());
    if (shiftYLeft==0) { bgrleftImage=tmpleftImage;  } else
                       {
                         tmpleftImage(cv::Rect(0,shiftYLeft, tmpleftImage.cols,tmpleftImage.rows-shiftYLeft)).copyTo(bgrleftImage(cv::Rect(0,0,tmpleftImage.cols,tmpleftImage.rows-shiftYLeft)));
                       }

    cv::Mat bgrrightImage = cv::Mat::zeros(tmprightImage.size(), tmprightImage.type());
    if (shiftYRight==0) { bgrrightImage=tmprightImage;  } else
                       {
                        tmprightImage(cv::Rect(0,shiftYRight, tmprightImage.cols,tmprightImage.rows-shiftYRight)).copyTo(bgrrightImage (cv::Rect(0,0,tmprightImage.cols,tmprightImage.rows-shiftYRight)));
                       }

    cv::Mat leftImage;
    cv::Mat rightImage;


    cv::cvtColor(bgrleftImage,leftImage, cv::COLOR_RGB2BGR);
    cv::cvtColor(bgrrightImage,rightImage, cv::COLOR_RGB2BGR);


    if(framesProcessed==0)
    {
      leftRGB=bgrleftImage;
      rightRGB=bgrrightImage;
      leftRGB.copyTo(lastLeftRGB);
      //lastLeftRGB=leftRGB;
      rightRGB.copyTo(lastRightRGB);
      //lastRightRGB=rightRGB;

      cv::cvtColor(bgrleftImage, greyLeft, CV_BGR2GRAY);
      cv::cvtColor(bgrrightImage, greyRight, CV_BGR2GRAY);
      greyLeft.copyTo(greyLastLeft);
      //greyLastLeft = greyLeft;
      greyRight.copyTo(greyLastRight);
      //greyLastRight = greyRight;

    } else
    {

      leftRGB.copyTo(lastLeftRGB);
      //lastLeftRGB=leftRGB;
      rightRGB.copyTo(lastRightRGB);
      //lastRightRGB=rightRGB;
      leftRGB=bgrleftImage;
      rightRGB=bgrrightImage;




      greyLeft.copyTo(greyLastLeft);
      //greyLastLeft = greyLeft;
      greyRight.copyTo(greyLastRight);
      //greyLastRight = greyRight;
      cv::cvtColor(bgrleftImage, greyLeft, CV_BGR2GRAY);
      cv::cvtColor(bgrrightImage, greyRight, CV_BGR2GRAY);

    }







    double alpha = 0.5;
    double beta = ( 1.0 - alpha );
    cv::Mat blend ;
    cv::addWeighted( leftImage, alpha, rightImage , beta, 0.0, blend);


    cv::Point pt1=cv::Point(0,0);
    cv::Point pt2=cv::Point(colorWidth,0);
    cv::Scalar color=cv::Scalar(0,255,0);
    unsigned int i=0;
    unsigned int blockY=(unsigned int) colorHeight/15;
    for (i=0; i<colorHeight/15; i++)
    {
       pt1.y=i*blockY; pt2.y=i*blockY;
       cv::line(blend,pt1,pt2,   color, 1, 8, 0);
    }

    cv::Point textPos=cv::Point(50,50);
    char outBuf[1024]={0};
    snprintf(outBuf,1024,"frame %u", framesProcessed );
    cv::putText( blend,outBuf, textPos , CV_FONT_HERSHEY_COMPLEX, 1 ,  cv::Scalar(0,255,0)  , 5, 2 );


    cv::imshow("blending",blend);
    std::cerr<<"Done with preliminary stuff..\n";
    //cv::waitKey(5);





    ++framesProcessed;
    return 1;
};
