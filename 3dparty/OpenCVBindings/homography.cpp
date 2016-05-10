
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include "homography.h"
#include "tools.h"



int checkHomographyFitness(
                           double thresholdX,double thresholdY ,
                           double * M ,
                           std::vector<cv::Point2f> &srcPoints ,
                           std::vector<cv::Point2f> &dstPoints ,
                           double * avgX ,   double * avgY ,
                           double * totAvgX , double * totAvgY ,
                           std::vector<cv::Point2f> &srcRANSACPoints ,
                           std::vector<cv::Point2f> &dstRANSACPoints
                          )
{
  srcRANSACPoints.clear();
  dstRANSACPoints.clear();
  *avgX = 0; *avgY = 0;

  // { { c a * sx + b sy - dx } ,  { f d sx + e sy - dy } } = 0
  unsigned int inlierCount=0;
  unsigned int i=0;

  double a=M[0] , b=M[1] , c=M[2];
  double d=M[3] , e=M[4] , f=M[5];
  double g=M[6] , h=M[7] , i2=M[8];

  double totSumX=0,totSumY=0;
  double sumX=0,sumY=0;
  double sx=0.0 , sy=0.0 ,  dx=0.0 , dy=0.0 , rx=0.0 , ry=0.0;




  for (i=0; i<srcPoints.size(); i++)
  {
    sx = (double) srcPoints[i].x;  sy = (double) srcPoints[i].y;
    dx = (double) dstPoints[i].x;  dy = (double) dstPoints[i].y;

    rx = (double) c + ( a * sx ) + ( b * sy )  ;
    ry = (double) f + ( d * sx ) + ( e * sy )  ;

    if ( rx > dx ) { rx = rx - dx; } else { rx = dx - rx; }
    if ( ry > dy ) { ry = ry - dy; } else { ry = dy - ry; }
    //fprintf(stderr," %0.2f %0.2f \n",rx,ry);

    totSumX+=rx; totSumY+=ry;


    if ( (rx< thresholdX )&&(ry< thresholdY ) )
    {
       sumX+=rx; sumY+=ry;
       ++inlierCount;

      cv::Point2f srcPT; srcPT.x = sx; srcPT.y = sy;
      cv::Point2f dstPT; dstPT.x = dx; dstPT.y = dy;
      srcRANSACPoints.push_back(srcPT);
      dstRANSACPoints.push_back(dstPT);
    }
  }

 if (inlierCount>0)
 {
  *avgX = (double) sumX/inlierCount;
  *avgY = (double) sumY/inlierCount;
  }

  *totAvgX=totSumX/srcPoints.size();
  *totAvgY=totSumY/srcPoints.size();

  //fprintf(stderr,"{ { a=%0.2f , b=%0.2f , c=%0.2f } , { d=%0.2f , e=%0.2f , f=%0.2f } }  ",a,b,c,d,e,f);
  //fprintf(stderr,"%u inliers / %u points  , average ( %0.2f,%0.2f ) \n",inlierCount , srcPoints.size() , *avgX , *avgY );
  return inlierCount;
}




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
 fprintf(stderr,"fitHomographyTransformationMatchesRANSAC start %0.2f \n",(float) srcPoints.size()/loops );
  if (srcPoints.size()<=4) { fprintf(stderr,"Cannot calculate a homography transformation without 4 or more point correspondances\n"); return 0; }
  if ((float)srcPoints.size()/loops <= 1.0 ) { fprintf(stderr,"Too few loops of ransac for our problem \n");   }

  //
  // DST | x |  = M | a b c | * SRC | x |
  //     | y |      | d e f |       | y |
  //     | z |      | g h i |       | 1 |
  //{ {dx} , {dy } , { dz }  } = { { a ,b, c} , { d , e ,f  }  , { g , h ,i  }  } . { { sx } , { sy } , { 1 } }
  //
  //

  unsigned int ptA=0,ptB=0,ptC=0,ptD=0;


  std::vector<cv::Point2f> bestSrcRANSACPoints;
  std::vector<cv::Point2f> bestDstRANSACPoints;

  cv::Point2f srcQuad[4];
  cv::Point2f dstQuad[4];


  cv::Mat bestWarp_mat( 3, 3,  CV_64FC1  );
  double bestM[9]={0};
  unsigned int bestInliers=0;
  double totAvgX=0,totAvgY=0,avgX=0,avgY=0;
  double bestTotAvgX=66666,bestTotAvgY=66666,bestAvgX=66660.0,bestAvgY=66660.0;

  unsigned int i=0,z=0;
  for (i=0; i<loops; i++)
  {
    //RANSAC pick 3 points
    ptA=rand()%srcPoints.size();
    ptB=rand()%srcPoints.size();
    ptC=rand()%srcPoints.size();
    ptD=rand()%srcPoints.size();
    do { ptB=rand()%srcPoints.size(); } while (ptB==ptA);
    do { ptC=rand()%srcPoints.size(); } while ( (ptC==ptA)||(ptC==ptB) );
    do { ptD=rand()%srcPoints.size(); } while ( (ptD==ptA)||(ptD==ptB)||(ptD==ptC) );

    dstQuad[0].x = dstPoints[ptA].x; dstQuad[0].y = dstPoints[ptA].y;
    dstQuad[1].x = dstPoints[ptB].x; dstQuad[1].y = dstPoints[ptB].y;
    dstQuad[2].x = dstPoints[ptC].x; dstQuad[2].y = dstPoints[ptC].y;
    dstQuad[3].x = dstPoints[ptD].x; dstQuad[3].y = dstPoints[ptD].y;

    srcQuad[0].x = srcPoints[ptA].x; srcQuad[0].y = srcPoints[ptA].y;
    srcQuad[1].x = srcPoints[ptB].x; srcQuad[1].y = srcPoints[ptB].y;
    srcQuad[2].x = srcPoints[ptC].x; srcQuad[2].y = srcPoints[ptC].y;
    srcQuad[3].x = srcPoints[ptD].x; srcQuad[3].y = srcPoints[ptD].y;

   /// Get the Affine Transform
   //derive resultMatrix with SVD
   warp_mat = cv::getPerspectiveTransform( srcQuad, dstQuad );
   //fprintf(stderr," ____________________________________________________________ \n");
   //std::cout << warp_mat<<"\n";

   M[0] = warp_mat.at<double>(0,0); M[1] = warp_mat.at<double>(0,1); M[2] = warp_mat.at<double>(0,2);
   M[3] = warp_mat.at<double>(1,0); M[4] = warp_mat.at<double>(1,1); M[5] = warp_mat.at<double>(1,2);
   M[6] = warp_mat.at<double>(2,0); M[7] = warp_mat.at<double>(2,1); M[8] = warp_mat.at<double>(2,2);


   unsigned int inliers = checkHomographyFitness( thresholdX , thresholdY , M , srcPoints , dstPoints , &avgX, &avgY , &totAvgX , &totAvgY , srcRANSACPoints , dstRANSACPoints );

   if (inliers > bestInliers )
    {
      bestInliers = inliers;
      bestAvgX = avgX;     bestAvgY = avgY;
      bestTotAvgX=totAvgX; bestTotAvgY=totAvgY;
      for (z=0; z<6; z++) { bestM[z]=M[z]; }
      bestSrcRANSACPoints=srcRANSACPoints;
      bestDstRANSACPoints=dstRANSACPoints;
      bestWarp_mat = warp_mat;
    }


    clear_line();
    fprintf(stderr,"RANSACing affine transform , %u / %u loops : \n",i,loops);
    fprintf(stderr,"Best : %u inliers  , avg ( %0.2f , %0.2f ) , global ( %0.2f , %0.2f )  \n",bestInliers,bestAvgX,bestAvgY , bestTotAvgX , bestTotAvgY);


    for (z=0; z<6; z++) { M[z]=bestM[z]; }
    srcRANSACPoints=bestSrcRANSACPoints;
    dstRANSACPoints= bestDstRANSACPoints;
    warp_mat = bestWarp_mat;

  }

  //fprintf(stderr,"fitAffineTransformationMatchesRANSAC done got %u inliers with avg ( %0.2f , %0.2f ) \n",bestInliers,bestAvgX,bestAvgY);
  return 1;
}
