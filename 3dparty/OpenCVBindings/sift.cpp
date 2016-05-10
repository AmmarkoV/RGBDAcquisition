#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <iostream>
#include <stdio.h>
#include "affine.h"
#include "tools.h"

const double THRESHOLD = 350;



double sumDistanceOfAllDescriptors(cv::Mat& vec1, cv::Mat& vec2)
{
  double sum = 0.0 , diff = 0.0;
  for (int i = 0; i < vec1.cols; i++)
  {
    diff = (vec1.at<uchar>(0,i) - vec2.at<uchar>(0,i)) ;
    sum +=  diff * diff;
  }
  return sqrt(sum);
}

int nearestNeighbor(cv::Mat& vec, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
  int neighbor = -1;
  double minDist = THRESHOLD+1;

  for (int i = 0; i < descriptors.rows; i++)
  {
    cv::KeyPoint pt = keypoints[i];
    cv::Mat v = descriptors.row(i);
    double d = sumDistanceOfAllDescriptors(vec, v);
    //printf("%d %f\n", v.cols, d);
    if (d < minDist) { minDist = d; neighbor = i; }
  }

  if (minDist < THRESHOLD) { return neighbor; } else
                           { /*Failed */}

  return -1;
}

void findPairs(std::vector<cv::KeyPoint>& keypoints1, cv::Mat& descriptors1,
               std::vector<cv::KeyPoint>& keypoints2, cv::Mat& descriptors2,
               std::vector<cv::Point2f>& srcPoints, std::vector<cv::Point2f>& dstPoints)
{
  for (int i = 0; i < descriptors1.rows; i++)
  {
    clear_line();
    fprintf(stderr,"Checking SIFT pairs , threshold %0.2f : \n",THRESHOLD);
    fprintf(stderr,"%u / %u checks  - %u matches \n",i, descriptors1.rows , srcPoints.size());
    cv::KeyPoint pt1 = keypoints1[i];
    cv::Mat desc1 = descriptors1.row(i);
    int nn = nearestNeighbor(desc1, keypoints2, descriptors2);
    if (nn >= 0) {
      cv::KeyPoint pt2 = keypoints2[nn];
      srcPoints.push_back(pt1.pt);
      dstPoints.push_back(pt2.pt);
    }
  }
}




