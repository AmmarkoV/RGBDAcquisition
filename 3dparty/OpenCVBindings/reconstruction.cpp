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








static void getCVMat3x3(double * M ,  cv::Mat & MCV)
{
 M[0] = MCV.at<double>(0,0); M[1] = MCV.at<double>(0,1); M[2] = MCV.at<double>(0,2);
 M[3] = MCV.at<double>(1,0); M[4] = MCV.at<double>(1,1); M[5] = MCV.at<double>(1,2);
 M[6] = MCV.at<double>(2,0); M[7] = MCV.at<double>(2,1); M[8] = MCV.at<double>(2,2);
}




int drawDepths(const char * filenameSrc , const char * filenameDst  , cv::Mat srcImg , cv::Mat dstImg , struct Point2DCorrespondance * correspondances )
{
  cv::Mat depthSrcImg = srcImg.clone();
  cv::Mat depthDstImg = dstImg.clone();

  cv::rectangle( depthSrcImg,
                 cv::Point( 0  , 0 ),
                 cv::Point( depthSrcImg.cols  , depthSrcImg.rows ),
                 cv::Scalar( 0, 0, 0 ),
                 -1,
                 8
              );

  cv::rectangle( depthDstImg,
                 cv::Point( 0  , 0 ),
                 cv::Point( depthDstImg.cols  , depthDstImg.rows ),
                 cv::Scalar( 0, 0, 0 ),
                 -1,
                 8
              );

  for (unsigned int i=0; i<correspondances->listCurrent; i++)
  {
      cv::rectangle( depthSrcImg,
                     cv::Point( correspondances->listSource[i].x-s  , correspondances->listSource[i].y-s ),
                     cv::Point( correspondances->listSource[i].x+s  , correspondances->listSource[i].y+s ),
                     cv::Scalar( correspondances->depth[i].x , correspondances->depth[i].x, correspondances->depth[i].x ),
                     -1,
                     8 );

      cv::rectangle( depthDstImg,
                     cv::Point( correspondances->listTarget[i].x-s  , correspondances->listTarget[i].y-s ),
                     cv::Point( correspondances->listTarget[i].x+s  , correspondances->listTarget[i].y+s ),
                     cv::Scalar( correspondances->depth[i].x , correspondances->depth[i].x, correspondances->depth[i].x ),
                     -1,
                     8 );

  }
 cv::imwrite(filenameSrc  , depthSrcImg);
 cv::imwrite(filenameDst  , depthDstImg);
}



int drawFeatures(const char * filenameSrc , const char * filenameDst , cv::Mat srcImg , cv::Mat dstImg , struct Point2DCorrespondance * correspondances )
{
  cv::Mat featureSrcImg = srcImg.clone();
  cv::Mat featureDstImg = dstImg.clone();

  for (unsigned int i=0; i<correspondances->listCurrent; i++)
  {
      cv::rectangle( featureSrcImg,
                     cv::Point( correspondances->listSource[i].x-s  , correspondances->listSource[i].y-s ),
                     cv::Point( correspondances->listSource[i].x+s  , correspondances->listSource[i].y+s ),
                     cv::Scalar( 0, 255, 255 ),
                     -1,
                     8 );


      cv::rectangle( featureDstImg,
                     cv::Point( correspondances->listTarget[i].x-s  , correspondances->listTarget[i].y-s ),
                     cv::Point( correspondances->listTarget[i].x+s  , correspondances->listTarget[i].y+s ),
                     cv::Scalar( 0, 255, 255 ),
                     -1,
                     8 );
  }
 cv::imwrite(filenameSrc  , featureSrcImg);
 cv::imwrite(filenameDst  , featureDstImg);
}

int drawEpipolarLines(const char * filename , cv::Mat img , std::vector<cv::Point3f> epilines)
{
  cv::Mat epipoleImg = img.clone();
  cv::Scalar color(256,0,256);

  for(size_t i=0; i<epilines.size(); i++)
  {
    cv::Point startPx(0,-epilines[i].z/epilines[i].y);
    cv::Point endPx(img.cols,-(epilines[i].z+epilines[i].x*img.cols)/epilines[i].y);

    cv::line(epipoleImg,startPx,endPx,color);
  }

 cv::imwrite(filename  , epipoleImg);
}







void triangulateFrom2HomographiesPre(struct Point2DCorrespondance * correspondances , double * H1 , double * H2)
{
  double x1 , y1 , x2 , y2 ;
  double triAX , triAY , triAZ;
  double triBX , triBY , triBZ;

  for (unsigned int i=0; i<correspondances->listCurrent; i++)
  {
   x1 = correspondances->listSource[i].x;
   y1 = correspondances->listSource[i].y;
   triAX = H1[6] + H1[0]*x1 + H1[3]*y1;
   triAY = H1[7] + H1[1]*x1 + H1[4]*y1;
   triAZ = H1[8] + H1[2]*x1 + H1[5]*y1;
   if (triAZ!=0.0)  { triAX=triAX/triAZ; triAY=triAY/triAZ; triAZ=1.0; }

   x2 = correspondances->listTarget[i].x;
   y2 = correspondances->listTarget[i].y;
   triBX = H2[6] + H2[0]*x2 + H2[3]*y2;
   triBY = H2[7] + H2[1]*x2 + H2[4]*y2;
   triBZ = H2[8] + H2[2]*x2 + H2[5]*y2;
   if (triBZ!=0.0)  { triBX=triBX/triBZ; triBY=triBY/triBZ; triBZ=1.0; }

   correspondances->depth[i].x = fabs((double) triBX-triAX);
   correspondances->depth[i].y = fabs((double) triBY-triAY);

   fprintf(stderr,"triangulateFrom2HomographiesPre %u - %0.2f/%0.2f \n", i , correspondances->depth[i].x , correspondances->depth[i].y );
  }
}



void triangulateFrom2HomographiesPost(struct Point2DCorrespondance * correspondances , double * H1 , double * H2)
{
  fprintf(stderr,"triangulateFrom2HomographiesPost  \n");
  double x1 , y1 , x2 , y2 ;
  double triAX , triAY , triAZ;
  double triBX , triBY , triBZ;

  for (unsigned int i=0; i<correspondances->listCurrent; i++)
  {
   x1 = correspondances->listSource[i].x;
   y1 = correspondances->listSource[i].y;
   triAX = H1[2] + H1[0]*x1 + H1[1]*y1;
   triAY = H1[5] + H1[3]*x1 + H1[4]*y1;
   triAZ = H1[8] + H1[6]*x1 + H1[7]*y1;
   if (triAZ!=0.0)  { triAX=triAX/triAZ; triAY=triAY/triAZ; triAZ=1.0; }
   fprintf(stderr,"%u - A(%0.2f,%0.2f,%0.2f)  ", i , triAX,triAY,triAZ);

   x2 = correspondances->listTarget[i].x;
   y2 = correspondances->listTarget[i].y;
   triBX = H1[2] + H1[0]*x2 + H1[1]*y2;
   triBY = H1[5] + H1[3]*x2 + H1[4]*y2;
   triBZ = H1[8] + H1[6]*x2 + H1[7]*y2;
   if (triBZ!=0.0)  { triBX=triBX/triBZ; triBY=triBY/triBZ; triBZ=1.0; }
   fprintf(stderr," B(%0.2f,%0.2f,%0.2f) ", triBX,triBY,triBZ);

   correspondances->depth[i].x = fabs((double) triBX-triAX);
   correspondances->depth[i].y = fabs((double) triBY-triAY);

   fprintf(stderr," - %0.2f/%0.2f \n", correspondances->depth[i].x , correspondances->depth[i].y );
  }
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

      correspondances->depth[i].x = distance3D( correspondances->listSource[i].x ,correspondances->listSource[i].y , 0.0 ,   correspondances->listTarget[i].x , correspondances->listTarget[i].y , 0.0 );
      correspondances->depth[i].y = 0;

  }

   drawFeatures( "rec1.jpg","rec2.jpg",image1 , image2 , correspondances );
   drawDepths( "naiveDepth1.jpg" , "naiveDepth2.jpg" , image1 , image2  , correspondances );

   cv::Mat fundMatCV( 3, 3,  CV_64FC1  );
   double fundMat[9]={0};
   std::vector<cv::Point2f> srcRANSACPoints;
   std::vector<cv::Point2f> dstRANSACPoints;

  if (useOpenCVEstimator)
  {
    fundMatCV = findFundamentalMat( srcPoints , dstPoints ,CV_FM_8POINT);
    fprintf(stderr,"Fundamental Matrix OpenCV: \n");
    std::cout << fundMatCV<<"\n";
    getCVMat3x3(fundMat,fundMatCV);

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
  cv::Mat H1CV(3,3, CV_64FC1);
  double H1[9]={0};
  cv::Mat H2CV(3,3, CV_64FC1);
  double H2[9]={0};

   if (useOpenCVEstimator)
  {
    cv::stereoRectifyUncalibrated(dstPoints, srcPoints, fundMatCV , image2.size(), H1CV, H2CV);

    fprintf(stderr,"Homography 1 : \n");
    std::cout << H1CV<<"\n";
    getCVMat3x3(H1,H1CV);


    fprintf(stderr,"Homography 2 : \n");
    std::cout << H2CV<<"\n";
    getCVMat3x3(H2,H2CV);


   cv::Size sz = cv::Size(image2.size().width  , image2.size().height );
   cv::Mat rectifiedLeft = cv::Mat::zeros(sz, CV_8UC3);
   cv::Mat rectifiedRight = cv::Mat::zeros(sz, CV_8UC3);
   //cv::warpAffine( image1,  rectifiedLeft   , H1CV,   rectifiedLeft.size() );
   cv::warpPerspective( image1, rectifiedLeft , H1CV , rectifiedLeft.size() );


   //cv::warpAffine( image2,  rectifiedRight   , H2CV,   rectifiedRight.size() );
   cv::warpPerspective( image2, rectifiedRight , H2CV , rectifiedRight.size() );

   cv::imwrite("rectified1.jpg", rectifiedLeft);
   cv::imwrite("rectified2.jpg", rectifiedRight);

   triangulateFrom2HomographiesPost( correspondances , H1 , H2);
   drawDepths( "depth1.jpg" , "depth2.jpg" , image1 , image2  , correspondances );
  }



  return 1;
}



