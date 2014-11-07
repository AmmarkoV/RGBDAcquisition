#include <iostream>


#define USE_NEW_OPENCV_HEADERS 1

#if USE_NEW_OPENCV_HEADERS
 //#include <opencv2/opencv.hpp>
 #include <opencv2/imgproc/imgproc_c.h>
 #include <opencv2/legacy/legacy.hpp>
 #include "opencv2/highgui/highgui.hpp"
#else
 #include <cv.h>
 #include <cxcore.h>
 #include <highgui.h>
#endif

using namespace std;

int main(int argc, char *argv[])
{
    IplImage* pImg = cvLoadImage(argv[1]);
    if(pImg == NULL) { return 1; }

    cvNamedWindow("FullScreenViewer", CV_WINDOW_NORMAL);
    cvSetWindowProperty("FullScreenViewer", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
    cvShowImage("FullScreenViewer", pImg);

    cvWaitKey();
    cvReleaseImage(&pImg); // Do not forget to release memory.

  return 0;
}
