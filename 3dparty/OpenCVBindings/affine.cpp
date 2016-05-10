
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <stdio.h>
#include "affine.h"
#include "tools.h"

void checkAffineSolution ( double * outX ,double * outY , double * dstX,double * dstY , double * M , double * srcX, double * srcY)
{
  double a=M[0] , b=M[1] , c=M[2];
  double d=M[3] , e=M[4] , f=M[5];

  *outX = c + ( (double) a * *srcX ) + ((double) b * *srcY );
  *outY = f + ( (double) d * *srcX ) + ((double) e * *srcY );


   if ( *outX > *dstX ) { *outX = *outX - *dstX; } else { *outX = *dstX - *outX; }
   if ( *outY > *dstY ) { *outY = *outY - *dstY; } else { *outY = *dstY - *outY; }

}

int checkAffineFitness(
                         double * M ,
                         std::vector<cv::Point2f> &srcPoints ,
                         std::vector<cv::Point2f> &dstPoints ,
                         double * avgX ,
                         double * avgY ,
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


    if ( (rx<3)&&(ry<3) )
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

  //fprintf(stderr,"{ { a=%0.2f , b=%0.2f , c=%0.2f } , { d=%0.2f , e=%0.2f , f=%0.2f } }  ",a,b,c,d,e,f);
  //fprintf(stderr,"%u inliers / %u points  , average ( %0.2f,%0.2f ) \n",inlierCount , srcPoints.size() , *avgX , *avgY );
  return inlierCount;
}




int fitAffineTransformationMatchesRANSAC(
                                          unsigned int loops ,
                                          double * M ,
                                          cv::Mat & warp_mat ,
                                          std::vector<cv::Point2f> &srcPoints ,
                                          std::vector<cv::Point2f> &dstPoints ,
                                          std::vector<cv::Point2f> &srcRANSACPoints ,
                                          std::vector<cv::Point2f> &dstRANSACPoints
                                        )
{
  fprintf(stderr,"fitAffineTransformationMatchesRANSAC start %0.2f \n",(float) srcPoints.size()/loops );
  if (srcPoints.size()<=3) { fprintf(stderr,"Cannot calculate an affine transformation without 3 or more point correspondances\n"); return 0; }
  if ((float)srcPoints.size()/loops <= 1.0 ) { fprintf(stderr,"Too many loops of ransac for our problem \n");   }

  //
  // DST | x |  = M | a b c | * SRC | x |
  //     | y |      | d e f |       | y |
  //                                | 1 |
  //{ {dx} , {dy } } = { { a ,b, c} , { d , e ,f  } } . { { sx } , { sy } , { 1 } }
  //{ {dx} , {dy } } = { { c a * sx + b sy } ,  { f d sx + e sy } }
  // { { c a * sx + b sy - dx } ,  { f d sx + e sy - dy } } = 0
  //
  //

  unsigned int ptA=0,ptB=0,ptC=0;


  std::vector<cv::Point2f> bestSrcRANSACPoints;
  std::vector<cv::Point2f> bestDstRANSACPoints;

  cv::Point2f srcTri[3];
  cv::Point2f dstTri[3];


  cv::Mat bestWarp_mat( 2, 3,  CV_64FC1  );
  double bestM[6]={0};
  unsigned int bestInliers=0;
  double avgX,avgY,bestAvgX=66660.0,bestAvgY=66660.0;

  unsigned int i=0,z=0;
  for (i=0; i<loops; i++)
  {
    //RANSAC pick 3 points
    ptA=rand()%srcPoints.size();
    ptB=rand()%srcPoints.size();
    ptC=rand()%srcPoints.size();
    do { ptB=rand()%srcPoints.size(); } while (ptB==ptA);
    do { ptC=rand()%srcPoints.size(); } while ( (ptC==ptA)||(ptB==ptC) );

    dstTri[0].x = dstPoints[ptA].x; dstTri[0].y = dstPoints[ptA].y;
    dstTri[1].x = dstPoints[ptB].x; dstTri[1].y = dstPoints[ptB].y;
    dstTri[2].x = dstPoints[ptC].x; dstTri[2].y = dstPoints[ptC].y;
    srcTri[0].x = srcPoints[ptA].x; srcTri[0].y = srcPoints[ptA].y;
    srcTri[1].x = srcPoints[ptB].x; srcTri[1].y = srcPoints[ptB].y;
    srcTri[2].x = srcPoints[ptC].x; srcTri[2].y = srcPoints[ptC].y;

   /// Get the Affine Transform
   //derive resultMatrix with Gauss Jordan
   warp_mat = cv::getAffineTransform( srcTri, dstTri );
   //fprintf(stderr," ____________________________________________________________ \n");
   //std::cout << warp_mat<<"\n";

   M[0] = warp_mat.at<double>(0,0); M[1] = warp_mat.at<double>(0,1); M[2] = warp_mat.at<double>(0,2);
   M[3] = warp_mat.at<double>(1,0); M[4] = warp_mat.at<double>(1,1); M[5] = warp_mat.at<double>(1,2);

  /*
   for (z=0; z<3; z++)
   {
    fprintf(stderr,"{ { %0.2f } , { %0.2f } }  = ",dstTri[z].x,dstTri[z].y);
    fprintf(stderr,"{ { %0.2f , %0.2f , %0.2f } , { %0.2f , %0.2f , %0.2f } }  ",M[0],M[1],M[2],M[3],M[4],M[5]);
    fprintf(stderr," . { { %0.2f } , { %0.2f } , { 1 } }  \n\n",srcTri[z].x,srcTri[z].y);


     double outX , outY , dstX = dstTri[z].x , dstY = dstTri[z].y , srcX = srcTri[z].x , srcY= srcTri[z].y;
     checkAffineSolution ( &outX , &outY , &dstX, &dstY , M , &srcX, &srcY);

     fprintf(stderr,"diff is %0.2f , %0.2f\n",outX , outY);
   }
  */


   unsigned int inliers = checkAffineFitness( M , srcPoints , dstPoints , &avgX, &avgY , srcRANSACPoints , dstRANSACPoints );

   if (inliers > bestInliers )
    {
      bestInliers = inliers;
      bestAvgX = avgX;
      bestAvgY = avgY;
      for (z=0; z<6; z++) { bestM[z]=M[z]; }
      bestSrcRANSACPoints=srcRANSACPoints;
      bestDstRANSACPoints=dstRANSACPoints;
      bestWarp_mat = warp_mat;
    }


    clear_line();
    fprintf(stderr,"RANSACing affine transform , %u / %u loops : \n",i,loops);
    fprintf(stderr,"Best : %u inliers  , avg ( %0.2f , %0.2f )  \n",bestInliers,bestAvgX,bestAvgY);


    for (z=0; z<6; z++) { M[z]=bestM[z]; }
    srcRANSACPoints=bestSrcRANSACPoints;
    dstRANSACPoints= bestDstRANSACPoints;
    warp_mat = bestWarp_mat;


  }

  //fprintf(stderr,"fitAffineTransformationMatchesRANSAC done got %u inliers with avg ( %0.2f , %0.2f ) \n",bestInliers,bestAvgX,bestAvgY);
  return 1;
}
