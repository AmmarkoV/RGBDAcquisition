#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <stdio.h>
#include "affine.h"
#include "homography.h"
#include "sift.h"
#include "stitcher.h"
#include "tools.h"

int visualizeMatches(
                      const char * filenameOutput ,
                      cv::Mat & left ,
                      std::vector<cv::KeyPoint>&keypointsLeft,
                      cv::Mat &descriptorsLeft,
                      cv::Mat & right ,
                      std::vector<cv::KeyPoint> &keypointsRight,
                      cv::Mat &descriptorsRight ,
                      std::vector<cv::Point2f> &srcPoints ,
                      std::vector<cv::Point2f> &dstPoints ,
                      std::vector<cv::Point2f> &srcRANSACPoints ,
                      std::vector<cv::Point2f> &dstRANSACPoints
                    )
{

 // Create a image for displaying mathing keypoints

    cv::Size size = left.size();
    cv::Size sz = cv::Size(size.width + right.size().width, size.height + right.size().height);
    cv::Mat matchingImage = cv::Mat::zeros(sz, CV_8UC3);

    // Draw camera frame
    cv::Mat roi1 = cv::Mat(matchingImage, cv::Rect(0, 0, size.width, size.height));
    left.copyTo(roi1);
    // Draw original image
    cv::Mat roi2 = cv::Mat(matchingImage, cv::Rect(size.width, size.height, right.size().width, right.size().height));
    right.copyTo(roi2);

    //char text[256];
    //sprintf(text, "%zd/%zd keypoints matched.", srcPoints.size(), keypointsLeft.size());
    //putText(matchingImage, text, cv::Point(0, cvRound(size.height + 30)), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0,0,255));


    for (unsigned int i=0; i<keypointsLeft.size(); i++)
    {
      cv::KeyPoint kp = keypointsLeft[i];
      cv::circle(matchingImage, kp.pt, cvRound(kp.size*0.25), cv::Scalar(255,255,0), 1, 8, 0);
    }


    for (int i=0; i<keypointsRight.size(); i++)
    {
      cv::KeyPoint kp = keypointsRight[i];

      cv::Point2f rightPT = kp.pt;
      rightPT.x += size.width;
      rightPT.y += size.height;

      cv::circle(matchingImage, rightPT , cvRound(kp.size*0.25), cv::Scalar(255,255,0), 1, 8, 0);
    }


if (srcRANSACPoints.size()>0)
{    // Draw line between ransac neighbor pairs
    for (int i = 0; i < (int)srcRANSACPoints.size(); ++i)
    {
      cv::Point2f pt1 = srcRANSACPoints[i];
      cv::Point2f pt2 = dstRANSACPoints[i];
      cv::Point2f from = pt1;
      cv::Point2f to   = cv::Point(size.width + pt2.x, size.height + pt2.y);
      cv::line(matchingImage, from, to, cv::Scalar(0, 255, 255));
    }
} else
{
    // Draw line between nearest neighbor pairs
    for (unsigned int i = 0; i < (int)srcPoints.size(); ++i)
    {
      cv::Point2f pt1 = srcPoints[i];
      cv::Point2f pt2 = dstPoints[i];
      cv::Point2f from = pt1;
      cv::Point2f to   = cv::Point(size.width + pt2.x, size.height + pt2.y);
      cv::line(matchingImage, from, to, cv::Scalar(0, 0 , 255));
    }
}

    // Display mathing image
    cv::imwrite(filenameOutput , matchingImage);
    imshow("mywindow", matchingImage);
    //int c = cv::waitKey(0);
}






int  sift_affine(const char * filenameLeft , const char * filenameRight ,  double SIFTThreshold ,
                 unsigned int RANSACLoops ,
                 unsigned int stitchedBorder ,
                 double reprojectionThresholdX ,
                 double reprojectionThresholdY ,
                 unsigned int useOpenCVHomographyEstimator
                 )
{

    fprintf(stderr,"Running SIFT on %s / %s \n" , filenameLeft , filenameRight);

    cv::Mat left = cv::imread(filenameLeft  , CV_LOAD_IMAGE_COLOR);
    if(! left.data ) { fprintf(stderr,"Left Image missing \n"); return 1; }

    cv::Mat right = cv::imread(filenameRight, CV_LOAD_IMAGE_COLOR);
    if(! right.data ) { fprintf(stderr,"Right Image missing \n"); return 1; }


    cv::DescriptorExtractor* extractor = new cv::SiftDescriptorExtractor();
    cv::SiftFeatureDetector detector;
    std::vector<cv::KeyPoint> keypointsLeft;
    cv::Mat descriptorsLeft;
    detector.detect(left, keypointsLeft);
    extractor->compute(left, keypointsLeft, descriptorsLeft);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(left, keypointsLeft, output);
    cv::imwrite("sift_features_left.jpg", output);


    std::vector<cv::KeyPoint> keypointsRight;
    cv::Mat descriptorsRight;
    detector.detect(right, keypointsRight);
    extractor->compute(right, keypointsRight, descriptorsRight);
    cv::drawKeypoints(right, keypointsRight, output);
    cv::imwrite("sift_features_right.jpg", output);

    //fprintf(stderr,"SIFT features ready \n");


    std::vector<cv::Point2f> srcRANSACPoints;
    std::vector<cv::Point2f> dstRANSACPoints;


    std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> dstPoints;
    findPairs( SIFTThreshold , keypointsLeft, descriptorsLeft, keypointsRight, descriptorsRight, srcPoints, dstPoints);
    //printf("%zd keypoints are matched.\n", srcPoints.size());



   visualizeMatches(
                      "sift_initial_match.jpg" ,
                      left ,
                      keypointsLeft,
                      descriptorsLeft,
                      right ,
                      keypointsRight,
                      descriptorsRight,
                      srcPoints,
                      dstPoints,
                      srcRANSACPoints,
                      dstRANSACPoints
                    );


   cv::Mat warp_mat( 2, 3,  CV_64FC1  );
   double M[6]={0};
   fitAffineTransformationMatchesRANSAC( RANSACLoops , reprojectionThresholdX , reprojectionThresholdY , M , warp_mat, srcPoints , dstPoints ,  srcRANSACPoints, dstRANSACPoints);


   stitchAffineMatch(
                     "wrappedAffine.jpg"  ,
                     stitchedBorder,
                     left ,
                     right ,
                     warp_mat
                    );


   visualizeMatches(
                      "sift_affine_match.jpg" ,
                      left ,
                      keypointsLeft,
                      descriptorsLeft,
                      right ,
                      keypointsRight,
                      descriptorsRight,
                      srcPoints,
                      dstPoints ,
                      srcRANSACPoints,
                      dstRANSACPoints
                    );




   cv::Mat homo_mat( 3, 3,  CV_64FC1  );
   double H[9]={0};

   if (useOpenCVHomographyEstimator)
   {
    homo_mat = cv::findHomography(srcPoints , dstPoints , CV_RANSAC);
   } else
   {
    fitHomographyTransformationMatchesRANSAC( RANSACLoops , reprojectionThresholdX , reprojectionThresholdY , H , homo_mat, srcPoints , dstPoints ,  srcRANSACPoints, dstRANSACPoints);
   }


   stitchHomographyMatch(
                         "wrappedHomography.jpg"  ,
                         stitchedBorder,
                         left ,
                         right ,
                         homo_mat
                        );

   visualizeMatches(
                      "sift_homography_match.jpg" ,
                      left ,
                      keypointsLeft,
                      descriptorsLeft,
                      right ,
                      keypointsRight,
                      descriptorsRight,
                      srcPoints,
                      dstPoints ,
                      srcRANSACPoints,
                      dstRANSACPoints
                    );
}








int main(int argc, const char* argv[])
{

   double SIFTThreshold = 350.0;

   unsigned int RANSACLoops = 1000;
   unsigned int stitchedBorder=100;
   double reprojectionThresholdX = 3.0;
   double reprojectionThresholdY = 3.0;

   unsigned int useOpenCVHomographyEstimator=0;


   char filenameLeft[512]={"uttower_left.JPG"};
   char filenameRight[512]={"uttower_right.JPG"};

   if (argc>=3)
   {
     snprintf(filenameLeft,512,"%s",argv[1]);
     snprintf(filenameRight,512,"%s",argv[2]);
   }


   for (unsigned int i=0; i<argc; i++)
   {
    if (strcmp(argv[i],"-opencv")==0) {
                                        fprintf(stderr,"Using opencv homography estimator \n");
                                        useOpenCVHomographyEstimator=1;
                                       } else
    if (strcmp(argv[i],"-loops")==0) {
                                      RANSACLoops=atoi(argv[i+1]);
                                      fprintf(stderr,"Using %u RANSAC loops\n",RANSACLoops);
                                     } else
    if (strcmp(argv[i],"-SIFTthreshold")==0)
                                     {
                                       SIFTThreshold=(double) atof(argv[i+1]);
                                       fprintf(stderr,"Setting SIFT threshold to %0.2f\n",SIFTThreshold);
                                      } else
    if (strcmp(argv[i],"-ReprojectionThreshold")==0)
                                     {
                                       reprojectionThresholdX=(double)atof(argv[i+1]);
                                       reprojectionThresholdY=reprojectionThresholdX;
                                       fprintf(stderr,"Setting reprojection threshold to %0.2f,%0.2f \n",reprojectionThresholdX,reprojectionThresholdY);
                                      }
   }


   sift_affine(filenameLeft,filenameRight , SIFTThreshold , RANSACLoops , stitchedBorder ,
                 reprojectionThresholdX ,
                 reprojectionThresholdY ,
                 useOpenCVHomographyEstimator);

    return 0;
}
