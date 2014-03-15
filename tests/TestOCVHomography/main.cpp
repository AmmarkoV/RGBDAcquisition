#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <vector>
#include <string>
#include <fstream>
#include <iostream>


int main(int argc, char* argv[])
{

   std::vector<cv::Point2f> sourcePoints;
   std::vector<cv::Point2f> destinationPoints;

  float x,y;

  x=34; y=379;  sourcePoints.push_back(cv::Point2f(x,y));
  x=178;y=379;  sourcePoints.push_back(cv::Point2f(x,y));
  x=320;y=379;  sourcePoints.push_back(cv::Point2f(x,y));
  x=461;y=379;  sourcePoints.push_back(cv::Point2f(x,y));
  x=605;y=379;  sourcePoints.push_back(cv::Point2f(x,y));
  x=120;y=312;  sourcePoints.push_back(cv::Point2f(x,y));
  x=219;y=312;  sourcePoints.push_back(cv::Point2f(x,y));
  x=319;y=312;  sourcePoints.push_back(cv::Point2f(x,y));

  x=33; y=358;  destinationPoints.push_back(cv::Point2f(x,y));
  x=84; y=374;  destinationPoints.push_back(cv::Point2f(x,y));
  x=139;y=392;  destinationPoints.push_back(cv::Point2f(x,y));
  x=202;y=410;  destinationPoints.push_back(cv::Point2f(x,y));
  x=271;y=432;  destinationPoints.push_back(cv::Point2f(x,y));
  x=88; y=318;  destinationPoints.push_back(cv::Point2f(x,y));
  x=134;y=329;  destinationPoints.push_back(cv::Point2f(x,y));
  x=184;y=342;  destinationPoints.push_back(cv::Point2f(x,y));

  if(sourcePoints.size() != destinationPoints.size())
    {
      std::cerr << "There must be the same number of points in both files (since they are correspondences!)." << "Source has " << sourcePoints.size() << " while destinationPoints has " << destinationPoints.size() << std::endl;
      return -1;
    }

  //cv::Mat H = cv::findHomography(sourcePoints, destinationPoints);
  //cv::Mat H = cv::findHomography(sourcePoints, destinationPoints, CV_RANSAC);
  cv::Mat H = cv::findHomography(sourcePoints, destinationPoints, CV_LMEDS);



  std::cout << "H = "<< " \n"  << H << "\n";


  return 0;
}
