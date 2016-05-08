#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <stdio.h>

const double THRESHOLD = 400;

/**
 * Calculate euclid distance
 */
double euclidDistance(cv::Mat& vec1, cv::Mat& vec2) {
  double sum = 0.0;
  int dim = vec1.cols;
  for (int i = 0; i < dim; i++) {
    sum += (vec1.at<uchar>(0,i) - vec2.at<uchar>(0,i)) * (vec1.at<uchar>(0,i) - vec2.at<uchar>(0,i));
  }
  return sqrt(sum);
}

/**
 * Find the index of nearest neighbor point from keypoints.
 */
int nearestNeighbor(cv::Mat& vec, std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {
  int neighbor = -1;
  double minDist = 1e6;

  for (int i = 0; i < descriptors.rows; i++) {
    cv::KeyPoint pt = keypoints[i];
    cv::Mat v = descriptors.row(i);
    double d = euclidDistance(vec, v);
    //printf("%d %f\n", v.cols, d);
    if (d < minDist) {
      minDist = d;
      neighbor = i;
    }
  }

  if (minDist < THRESHOLD) {
    return neighbor;
  }

  return -1;
}

/**
 * Find pairs of points with the smallest distace between them
 */
void findPairs(std::vector<cv::KeyPoint>& keypoints1, cv::Mat& descriptors1,
               std::vector<cv::KeyPoint>& keypoints2, cv::Mat& descriptors2,
               std::vector<cv::Point2f>& srcPoints, std::vector<cv::Point2f>& dstPoints) {
  for (int i = 0; i < descriptors1.rows; i++)
  {
    fprintf(stderr,"%u / %u matches \n",i, descriptors1.rows);
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






int visualizeMatches(
                      cv::Mat & left ,
                      std::vector<cv::KeyPoint>&keypointsLeft,
                      cv::Mat &descriptorsLeft,
                      cv::Mat & right ,
                      std::vector<cv::KeyPoint> &keypointsRight,
                      cv::Mat &descriptorsRight ,
                      std::vector<cv::Point2f> &srcPoints ,
                      std::vector<cv::Point2f> &dstPoints
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

    char text[256];
    sprintf(text, "%zd/%zd keypoints matched.", srcPoints.size(), keypointsLeft.size());
    putText(matchingImage, text, cv::Point(0, cvRound(size.height + 30)), cv::FONT_HERSHEY_SCRIPT_SIMPLEX, 1, cv::Scalar(0,0,255));


    for (int i=0; i<keypointsLeft.size(); i++){
      cv::KeyPoint kp = keypointsLeft[i];
      cv::circle(matchingImage, kp.pt, cvRound(kp.size*0.25), cv::Scalar(255,255,0), 1, 8, 0);
    }

    // Draw line between nearest neighbor pairs
    for (int i = 0; i < (int)srcPoints.size(); ++i)
    {
      cv::Point2f pt1 = srcPoints[i];
      cv::Point2f pt2 = dstPoints[i];
      cv::Point2f from = pt1;
      cv::Point2f to   = cv::Point(size.width + pt2.x, size.height + pt2.y);
      cv::line(matchingImage, from, to, cv::Scalar(0, 255, 255));
    }

    // Display mathing image
    cv::imwrite("sift_result_match.jpg", matchingImage);
    imshow("mywindow", matchingImage);
    int c = cv::waitKey(0);
}






int main(int argc, const char* argv[])
{
    cv::Mat left = cv::imread("uttower_left.JPG", 0); //Load as grayscale
    cv::Mat right = cv::imread("uttower_right.JPG", 0); //Load as grayscale


    cv::DescriptorExtractor* extractor = new cv::SiftDescriptorExtractor();
    cv::SiftFeatureDetector detector;
    std::vector<cv::KeyPoint> keypointsLeft;
    cv::Mat descriptorsLeft;
    detector.detect(left, keypointsLeft);
    extractor->compute(left, keypointsLeft, descriptorsLeft);

    // Add results to image and save.
    cv::Mat output;
    cv::drawKeypoints(left, keypointsLeft, output);
    cv::imwrite("sift_result_left.jpg", output);


    std::vector<cv::KeyPoint> keypointsRight;
    cv::Mat descriptorsRight;
    detector.detect(right, keypointsRight);
    extractor->compute(right, keypointsRight, descriptorsRight);
    cv::drawKeypoints(right, keypointsRight, output);
    cv::imwrite("sift_result_right.jpg", output);

    fprintf(stderr,"SIFT features ready \n");




    std::vector<cv::Point2f> srcPoints;
    std::vector<cv::Point2f> dstPoints;
    findPairs(keypointsLeft, descriptorsLeft, keypointsRight, descriptorsRight, srcPoints, dstPoints);
    printf("%zd keypoints are matched.\n", srcPoints.size());



   visualizeMatches(
                      left ,
                      keypointsLeft,
                      descriptorsLeft,
                      right ,
                      keypointsRight,
                      descriptorsRight,
                      srcPoints,
                      dstPoints
                    );




    return 0;
}
