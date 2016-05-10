#include "homography.h"






int fitHomographyTransformationMatchesRANSAC(
                                             unsigned int loops ,
                                             double thresholdX,double thresholdY ,
                                             double * M ,
                                             cv::Mat & warp_mat ,
                                             std::vector<cv::Point2f> &srcPoints ,
                                             std::vector<cv::Point2f> &dstPoints ,
                                             std::vector<cv::Point2f> &srcRANSACPoints ,
                                             std::vector<cv::Point2f> &dstRANSACPoints
                                            )
{

  return 1;
}
