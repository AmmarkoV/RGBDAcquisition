#ifndef AFFINE_H_INCLUDED
#define AFFINE_H_INCLUDED

#include <opencv2/core/core.hpp>

int fitAffineTransformationMatchesRANSAC(
                                          unsigned int loops ,
                                          double thresholdX,double thresholdY ,
                                          double * M ,
                                          cv::Mat & warp_mat ,
                                          std::vector<cv::Point2f> &srcPoints ,
                                          std::vector<cv::Point2f> &dstPoints ,
                                          std::vector<cv::Point2f> &srcRANSACPoints ,
                                          std::vector<cv::Point2f> &dstRANSACPoints
                                        );

#endif // AFFINE_H_INCLUDED
