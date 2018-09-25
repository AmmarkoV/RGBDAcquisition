#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

using namespace cv;

void drawLine( cv::Mat img,
               unsigned int x1,unsigned int y1,
               unsigned int x2,unsigned int y2,
               unsigned int r,
               unsigned int g,
               unsigned int b
             )
{
  cv::Point start = cv::Point(x1,y1);
  cv::Point end = cv::Point(x2,y2);

  int thickness = 2;
  int lineType = 8;
  cv::line( img,
            start,
            end,
            cv::Scalar( b, g, r ),
            thickness,
            lineType );
}


int webcamProgram()
{


    VideoCapture cap;
    // open the default camera, use something different from 0 otherwise;
    // Check VideoCapture documentation.
    if(!cap.open(0))
        return 0;
    for(;;)
    {
          Mat frame;
          cap >> frame;
          if( frame.empty() ) break; // end of video stream
          imshow("this is you, smile! :)", frame);
          if( waitKey(10) == 27 ) break; // stop capturing by pressing ESC
    }
    // the camera will be closed automatically upon exit
    // cap.close();


}


int main (int argc,const char *argv[])
{
    cv::Mat image = cv::imread("lena.jpeg", CV_LOAD_IMAGE_COLOR);

    drawLine(image,30,30,40,40 , 255,0,0 );

    cv::imshow("Test",image);







    //webcamProgram();







    cv::waitKey(0);

    return 0;
}

