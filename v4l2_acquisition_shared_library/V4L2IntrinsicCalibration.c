#include "V4L2IntrinsicCalibration.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/*

   RectifiedPoint = M * OriginalPoint

   Rectified Point = [ new_x new_y new_w ]
   Original Point  = [ x y w ]

         |fx  0   cx|       a   b   c
   M =   |0   fy  cy|       d   e   f
         |0   0   1 |       g   h   i
*/
unsigned int PrecalcResectioning(unsigned int * frame ,     unsigned int width , unsigned int height ,
                                                            double fx,double fy , double cx,double cy ,
                                                            double k1,double k2 , double p1,double p2 , double k3   )
{
  if ( frame == 0 )
    {
       fprintf(stderr , "WARNING : PrecalcResectioning called with a zero frame to work on..!\n");
       fprintf(stderr , "WARNING : This means that precalculations haven't been made..!\n");
       fprintf(stderr , "WARNING : PrecalcResectioning code will now return without doing anything..\n");
       return 0;
    }


  fprintf(stderr,"Calibrating fx=%f fy=%f cx=%f cy=%f\n",fx,fy,cx,cy);
  fprintf(stderr,"k1=%f k2=%f p1=%f p2=%f k3=%f \n",k1,k2,p1,p2,k3);

  if ( ( fx == 0.0) || ( fy == 0.0) || ( (k1==0.0)&&(k2==0.0)&&(k3==0.0) )) { fprintf(stderr,"Erroneous parameters calibration canceled\n"); return 0; }

  unsigned int i,x =width ,y=height , mem , new_mem , interpolation_mem;
  unsigned int undistorted_x,undistorted_y;

  mem = 0;
  double ifx=1.f/fx,ify=1.f/fy;
  double dstdx,dstdy , distx,disty;
  double dx,dy;
  double r_sq  = 0;  // R Squared
  double r_cu = 0;   // R Cubed
  double k_coefficient = 0;
  double new_x,new_y;

  // SEE http://opencv.willowgarage.com/documentation/camera_calibration_and_3d_reconstruction.html
  // https://code.ros.org/trac/opencv/browser/trunk/opencv/src/cv/cvundistort.cpp?rev=18
  // https://code.ros.org/trac/opencv/browser/trunk/opencv/modules/imgproc/src/undistort.cpp?rev=4885
  // http://tech.groups.yahoo.com/group/OpenCV/message/26019
  // Also Learning OpenCV page 375-377
  /*

        Finaly fixed using code from Philip Gruebele @
            http://tech.groups.yahoo.com/group/OpenCV/message/26019

            archived at 3dpartylibs/code/undistort_point.cpp
  */
  unsigned int PrecisionErrors=0;
  unsigned int OffFrame=0;
  unsigned int OutOfMemory=0;


  for (y=0; y<height; y++)
  {
     interpolation_mem=0;
     for (x=0; x<width; x++)
        {
          //Well this is supposed to rectify lens distortions based on calibration done with my image sets
          //but the values returned are way off ..
          dstdx = ( x - cx );
          dstdx *=  ifx;

          dstdy = ( y - cy );
          dstdy *=  ify;

          new_x = dstdx;
          new_y = dstdy;

          for ( i=0; i<5; i++)
           {
               r_sq =  new_x*new_x;
               r_sq += new_y*new_y;

               r_cu = r_sq*r_sq;

               k_coefficient = 1;
               k_coefficient += k1 * r_sq;
               k_coefficient += k2 * r_cu;
               k_coefficient += k3 * r_cu * r_sq ;

               dx =  2 * p1 * new_x * new_y;
               dx += p2 * ( r_sq + 2 * new_x * new_x);

               dy =  2 * p2 * new_x * new_y;
               dy += p1 * ( r_sq + 2 * new_y * new_y);

               new_x = ( dstdx - dx );
               new_x /= k_coefficient;

               new_y = ( dstdy - dy );
               new_y /= k_coefficient;
           }

          dstdx = new_x;
          dstdy = new_y;

          dstdx *= fx; dstdx += cx;
          dstdy *= fy; dstdy += cy;


          undistorted_x  = (unsigned int) round(dstdx);
          undistorted_y  = (unsigned int) round(dstdy);



                   /* REVERSE CHECK ! >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>*/
                   new_x = dstdx - cx; new_x /= fx;
                   new_y = dstdy - cy; new_y /= fy;

                   r_sq = new_x*new_x + new_y*new_y;
                   distx = new_x ;
                   distx += new_x*(k1*r_sq + k2*r_sq*r_sq);
                   distx += (2*p1*new_x*new_y + p2*(r_sq + 2*new_x*new_x));


                   disty = new_y;
                   disty +=new_y*(k1*r_sq + k2*r_sq*r_sq);
                   disty +=(2*p2*new_x*new_y + p1*(r_sq + 2*new_y*new_y));


                   distx *= fx; distx += cx;
                   disty *= fy; disty += cy;
                   /* <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<*/



      // This should never be more than .2 pixels...
      double diffx = x - distx;
      double diffy = y - disty;

         if ( (diffx> 0.1) || (diffy>0.1) )
          {
             /* ACCURACY ERROR , This means that we have a percision error in the way math is done*/
             //fprintf(stderr,"$%u,%u to %u,%u",x,y,undistorted_x,undistorted_y);
             new_mem = 0;
                 new_mem=interpolation_mem ; //TEST THIS USESTHE INTERPOLATED TO GET RID OF SOME BLANK SPOT ARTIFACTS
             ++PrecisionErrors;
          }

          if ( ( undistorted_x >= width ) || ( undistorted_y >= height ) )
             {
                 // OFF RESULTS SHOULD BE INTERPOLATED WITH CLOSE MEMORY SPOTS
                 //fprintf(stderr,"!%u,%u to %u,%u",x,y,undistorted_x,undistorted_y);
                 new_mem = 0;
                 new_mem=interpolation_mem ; //TEST THIS USESTHE INTERPOLATED TO GET RID OF SOME BLANK SPOT ARTIFACTS
                 ++OffFrame;
             } else
             {
                new_mem = undistorted_y * (width*3) + undistorted_x * 3 ;
                interpolation_mem = new_mem;
                if ( new_mem>= width*height*3 )
                 {
                   new_mem = 0;
                   ++OutOfMemory;
                  }
                //fprintf(stderr,"%u,%u -> %u,%u .. \n",x,y,undistorted_x,undistorted_y);
             }



          frame [mem] = new_mem;
          ++mem;  ++new_mem;

          frame [mem] = new_mem;
          ++mem;  ++new_mem;

          frame [mem] = new_mem;
          ++mem;  ++new_mem;

       }
   }

 fprintf(stderr,"PrecalculationErrors - Precision=%u , OffFrame=%u , OutOfMemory=%u\n",PrecisionErrors,OffFrame,OutOfMemory);

 return 1;

}
