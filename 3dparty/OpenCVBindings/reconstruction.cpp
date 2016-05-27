#include "reconstruction.h"
#include "homography.h"
#include "fundamental.h"
#include "primitives.h"


#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <opencv2/calib3d/calib3d.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>



  unsigned int s=2;

static double distance3D(double p1X , double p1Y  , double p1Z ,  double p2X , double p2Y , double p2Z)
{
  double vect_x = p1X - p2X;
  double vect_y = p1Y - p2Y;
  double vect_z = p1Z - p2Z;
  double len = sqrt( vect_x*vect_x + vect_y*vect_y + vect_z*vect_z);
  if(len == 0) len = 1.0f;
return len;
}




int drawFeatures(cv::Mat srcImg , cv::Mat dstImg , struct Point2DCorrespondance * correspondances )
{
  for (unsigned int i=0; i<correspondances->listCurrent; i++)
  {
      cv::rectangle( srcImg,
                     cv::Point( correspondances->listSource[i].x-s  , correspondances->listSource[i].y-s ),
                     cv::Point( correspondances->listSource[i].x+s  , correspondances->listSource[i].y+s ),
                     cv::Scalar( 0, 255, 255 ),
                     -1,
                     8 );


      cv::rectangle( dstImg,
                     cv::Point( correspondances->listTarget[i].x-s  , correspondances->listTarget[i].y-s ),
                     cv::Point( correspondances->listTarget[i].x+s  , correspondances->listTarget[i].y+s ),
                     cv::Scalar( 0, 255, 255 ),
                     -1,
                     8 );
  }
}

int drawEpipolarLines(const char * filename , cv::Mat img , std::vector<cv::Point3f> epilines)
{
  cv::Mat epipoleImg = img;
  cv::Scalar color(256,0,256);

  for(size_t i=0; i<epilines.size(); i++)
  {
    cv::Point startPx(0,-epilines[i].z/epilines[i].y);
    cv::Point endPx(img.cols,-(epilines[i].z+epilines[i].x*img.cols)/epilines[i].y);

    cv::line(epipoleImg,startPx,endPx,color);
  }

 cv::imwrite(filename  , epipoleImg);
}


int  reconstruct3D(const char * filenameLeft , unsigned int useOpenCVEstimator )
{
  char filename[512]={0};

  snprintf(filename,512,"%s/%s1.jpg",filenameLeft ,filenameLeft);
  cv::Mat image1 = cv::imread(filename  , CV_LOAD_IMAGE_COLOR);
  if(! image1.data ) { fprintf(stderr,"Image1 missing \n"); return 0; }

  snprintf(filename,512,"%s/%s2.jpg",filenameLeft ,filenameLeft);
  cv::Mat image2 = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  if(! image2.data ) { fprintf(stderr,"Image2 missing \n"); return 0; }

  cv::Mat imageOutput = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  cv::rectangle( imageOutput,
                 cv::Point( 0  , 0 ),
                 cv::Point( imageOutput.cols  , imageOutput.rows ),
                 cv::Scalar( 0, 0, 0 ),
                 -1,
                 8
              );

  std::vector<cv::Point2f> srcPoints;
  std::vector<cv::Point2f> dstPoints;

  snprintf(filename,512,"%s/%s_matches.txt",filenameLeft ,filenameLeft);
  struct Point2DCorrespondance * correspondances = readPointList( filename );
  for (unsigned int i=0; i<correspondances->listCurrent; i++)
  {

      printf("#%u (%0.2f , %0.2f ) -> ( %0.2f , %0.2f) \n" , i ,
        correspondances->listSource[i].x ,
        correspondances->listSource[i].y ,
        correspondances->listTarget[i].x ,
        correspondances->listTarget[i].y  );



      cv::Point2f srcPT; srcPT.x = correspondances->listSource[i].x; srcPT.y = correspondances->listSource[i].y;
      cv::Point2f dstPT; dstPT.x = correspondances->listTarget[i].x; dstPT.y = correspondances->listTarget[i].y;

      srcPoints.push_back(srcPT);
      dstPoints.push_back(dstPT);

    double depth = distance3D( correspondances->listSource[i].x ,correspondances->listSource[i].y , 0.0 ,   correspondances->listTarget[i].x , correspondances->listTarget[i].y , 0.0 );

      cv::rectangle( imageOutput,
                     cv::Point( correspondances->listTarget[i].x-s  , correspondances->listTarget[i].y-s ),
                     cv::Point( correspondances->listTarget[i].x+s  , correspondances->listTarget[i].y+s ),
                     cv::Scalar( depth, depth, depth ),
                     -1,
                     8 );
  }

   drawFeatures( image1 , image2 , correspondances );



   cv::Mat fundMatCV( 3, 3,  CV_64FC1  );
   double fundMat[9]={0};
    std::vector<cv::Point2f> srcRANSACPoints;
    std::vector<cv::Point2f> dstRANSACPoints;

  if (useOpenCVEstimator)
  {
    fundMatCV = findFundamentalMat( srcPoints , dstPoints ,CV_FM_8POINT);
    fprintf(stderr,"Fundamental Matrix OpenCV: \n");
     std::cout << fundMatCV<<"\n";

   fundMat[0] = fundMatCV.at<double>(0,0); fundMat[1] = fundMatCV.at<double>(0,1); fundMat[2] = fundMatCV.at<double>(0,2);
   fundMat[3] = fundMatCV.at<double>(1,0); fundMat[4] = fundMatCV.at<double>(1,1); fundMat[5] = fundMatCV.at<double>(1,2);
   fundMat[6] = fundMatCV.at<double>(2,0); fundMat[7] = fundMatCV.at<double>(2,1); fundMat[8] = fundMatCV.at<double>(2,2);

   fprintf(stderr,"Fundamental Matrix Copy : \n");
   for (unsigned int i=0; i<9; i+=3)
    { fprintf(stderr," %0.2f %0.2f %0.2f  \n", fundMat[i+0], fundMat[i+1], fundMat[i+2] ); }

  } else
  {
   fitHomographyTransformationMatchesRANSAC(
                                             1000 ,
                                             5.0 , 5.0 ,
                                             fundMat ,
                                             fundMatCV ,
                                             srcPoints ,
                                             dstPoints ,
                                             srcRANSACPoints ,
                                             dstRANSACPoints
                                            );

   fprintf(stderr,"Fundamental Matrix Mine : \n");
   for (unsigned int i=0; i<9; i+=3)
    { fprintf(stderr," %0.2f %0.2f %0.2f  \n", fundMat[i+0], fundMat[i+1], fundMat[i+2] ); }
  }

   testAverageFundamentalMatrixForAllPairs(correspondances , fundMat );



  std::vector<cv::Point3f> epilines1 , epilines2 ;
   cv::computeCorrespondEpilines(srcPoints,1,fundMatCV, epilines1);
   cv::computeCorrespondEpilines(dstPoints,2,fundMatCV, epilines2);


    cv::imwrite("rec1.jpg", image1);
    cv::imwrite("rec2.jpg", image2);
    cv::imwrite("recDepth.jpg", imageOutput);

   drawEpipolarLines("epipoles1.jpg",image1 , epilines1);
   drawEpipolarLines("epipoles2.jpg",image2 , epilines2);




  double camera1Mat[16]={0};
  snprintf(filename,512,"%s/%s1_camera.txt",filenameLeft ,filenameLeft);
  struct Point2DCorrespondance * camera1 = readPointList( filename );
  camera1Mat[0] = camera1->listSource[0].x; camera1Mat[1] = camera1->listSource[0].y;
  camera1Mat[2] = camera1->listTarget[0].x; camera1Mat[3] = camera1->listTarget[0].y;

  camera1Mat[4] = camera1->listSource[1].x; camera1Mat[5] = camera1->listSource[1].y;
  camera1Mat[6] = camera1->listTarget[1].x; camera1Mat[7] = camera1->listTarget[1].y;

  camera1Mat[8] = camera1->listSource[2].x; camera1Mat[9] = camera1->listSource[2].y;
  camera1Mat[10]= camera1->listTarget[2].x; camera1Mat[11]= camera1->listTarget[2].y;

  fprintf(stderr,"Camera1 : \n");
  for (unsigned int i=0; i<14; i+=4)
   { fprintf(stderr," %0.2f %0.2f %0.2f %0.2f \n", camera1Mat[i+0], camera1Mat[i+1], camera1Mat[i+2], camera1Mat[i+3]); }


  double camera2Mat[16]={0};
  snprintf(filename,512,"%s/%s2_camera.txt",filenameLeft ,filenameLeft);
  struct Point2DCorrespondance * camera2 = readPointList( filename );
  camera2Mat[0] = camera2->listSource[0].x; camera2Mat[1] = camera2->listSource[0].y;
  camera2Mat[2] = camera2->listTarget[0].x; camera2Mat[3] = camera2->listTarget[0].y;

  camera2Mat[4] = camera2->listSource[1].x; camera2Mat[5] = camera2->listSource[1].y;
  camera2Mat[6] = camera2->listTarget[1].x; camera2Mat[7] = camera2->listTarget[1].y;

  camera2Mat[8] = camera2->listSource[2].x; camera2Mat[9] = camera2->listSource[2].y;
  camera2Mat[10]= camera2->listTarget[2].x; camera2Mat[11]= camera2->listTarget[2].y;


  fprintf(stderr,"Camera2 : \n");
  for (unsigned int i=0; i<14; i+=4)
   { fprintf(stderr," %0.2f %0.2f %0.2f %0.2f \n", camera2Mat[i+0], camera2Mat[i+1], camera2Mat[i+2], camera2Mat[i+3]); }



    //H1, H2 – The output rectification homography matrices for the first and for the second images.
    cv::Mat H1(4,4, CV_64FC1);
    cv::Mat H2(4,4, CV_64FC1);

   if (useOpenCVEstimator)
  {
    cv::stereoRectifyUncalibrated(dstPoints, srcPoints, fundMatCV , image2.size(), H1, H2);

    fprintf(stderr,"Homography 1 : \n");
    std::cout << H1<<"\n";

    fprintf(stderr,"Homography 2 : \n");
    std::cout << H2<<"\n";
  }



/*
16.11 13.70 -67.35 -188.38
 0.83 -61.26 -27.99 -7.42
 0.17 -0.05 -0.08 0.57
*/



  return 1;
}



