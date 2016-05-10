#ifndef SIFT_H_INCLUDED
#define SIFT_H_INCLUDED

#include <opencv2/core/core.hpp>

void findPairs(std::vector<cv::KeyPoint>& keypoints1, cv::Mat& descriptors1,
               std::vector<cv::KeyPoint>& keypoints2, cv::Mat& descriptors2,
               std::vector<cv::Point2f>& srcPoints, std::vector<cv::Point2f>& dstPoints);

#endif // SIFT_H_INCLUDED
