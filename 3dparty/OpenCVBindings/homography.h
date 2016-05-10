#ifndef HOMOGRAPHY_H_INCLUDED
#define HOMOGRAPHY_H_INCLUDED


#include <opencv2/core/core.hpp>

int fitHomographyTransformationMatchesRANSAC(
                                             unsigned int loops ,
                                             double thresholdX,double thresholdY ,
                                             double * M ,
                                             cv::Mat & warp_mat ,
                                             std::vector<cv::Point2f> &srcPoints ,
                                             std::vector<cv::Point2f> &dstPoints ,
                                             std::vector<cv::Point2f> &srcRANSACPoints ,
                                             std::vector<cv::Point2f> &dstRANSACPoints
                                            );

#endif // HOMOGRAPHY_H_INCLUDED
