#include "transform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../AmMatrix/matrix3x3Tools.h"


enum VecDims
{
  VEC_X = 0 ,
  VEC_Y  ,
  VEC_Z  ,
  VEC_W  ,
  VEC_DIMENSIONS
};




unsigned char *  registerUndistortedColorToUndistortedDepthFrame
                                          (
                                           unsigned char * undistortedRgb , unsigned int rgbWidth , unsigned int rgbHeight , struct calibration * rgbCalibration ,
                                           unsigned short * undistortedDepth , unsigned int depthWidth , unsigned int depthHeight , struct calibration * depthCalibration ,
                                           double * rotation3x3 , double * translation3x1 ,
                                           unsigned int * outputWidth , unsigned int * outputHeight
                                          )
{
  //This is nicely explained here : http://nicolas.burrus.name/index.php/Research/KinectCalibration
  /*
  P3D.x = (x_d - cx_d) * depth(x_d,y_d) / fx_d
  P3D.y = (y_d - cy_d) * depth(x_d,y_d) / fy_d
  P3D.z = depth(x_d,y_d)

   with fx_d, fy_d, cx_d and cy_d the intrinsics of the depth camera.

   We can then reproject each 3D point on the color image and get its color:

   P3D' = R.P3D + T
   P2D_rgb.x = (P3D'.x * fx_rgb / P3D'.z) + cx_rgb
   P2D_rgb.y = (P3D'.y * fy_rgb / P3D'.z) + cy_rgb
*/
  unsigned char * undistortedRgbPtr = undistortedRgb;
  unsigned int offsetRgb;

  unsigned short * undistortedDepthPtr = undistortedDepth;
  unsigned short * undistortedDepthLimit = undistortedDepth + (depthWidth * depthHeight * sizeof(unsigned short) );
  unsigned char  * output = (unsigned char * ) malloc(rgbWidth * rgbHeight * 3 * sizeof(char));
  if (output==0) { fprintf(stderr,"Could not allocate a place to store the registered feed"); return 0; }

  unsigned short depth_x_d_y_d=0;
  unsigned int x_d = 0 , y_d = 0;

  double P3D_B[VEC_DIMENSIONS]={0};
  double P3D_A[VEC_DIMENSIONS]={0};
  double *P3D_x=&P3D_A[VEC_X] , *P3D_y=&P3D_A[VEC_Y] , *P3D_z=&P3D_A[VEC_Z]; P3D_A[VEC_W]=1.0;

  double P2D_rgb_x, P2D_rgb_y ;

  double cx_d = depthCalibration->intrinsic[CALIB_INTR_CX];
  double cy_d = depthCalibration->intrinsic[CALIB_INTR_CY];
  double fx_d = depthCalibration->intrinsic[CALIB_INTR_FX];
  double fy_d = depthCalibration->intrinsic[CALIB_INTR_FY];


  double cx_rgb = rgbCalibration->intrinsic[CALIB_INTR_CX];
  double cy_rgb = rgbCalibration->intrinsic[CALIB_INTR_CY];
  double fx_rgb = rgbCalibration->intrinsic[CALIB_INTR_FX];
  double fy_rgb = rgbCalibration->intrinsic[CALIB_INTR_FY];

  unsigned char * outputPTR=output;
  while (undistortedDepthPtr<undistortedDepthLimit)
  {
   depth_x_d_y_d=(*undistortedDepthPtr);
   *P3D_x = (double) (x_d - cx_d) * depth_x_d_y_d / fx_d;
   *P3D_y = (double) (y_d - cy_d) * depth_x_d_y_d / fy_d;
   *P3D_z = (double) depth_x_d_y_d;

   transform2DPointVectorUsing3x3Matrix(P3D_B,rotation3x3,P3D_A);
   P3D_B[VEC_X]+=translation3x1[VEC_X];
   P3D_B[VEC_Y]+=translation3x1[VEC_Y];
   P3D_B[VEC_Z]+=translation3x1[VEC_Z];

   if (P3D_B[VEC_Z]!=0)
    {
     P2D_rgb_x = (P3D_B[VEC_X] * fx_rgb / P3D_B[VEC_Z]) + cx_rgb;
     P2D_rgb_y = (P3D_B[VEC_Y] * fy_rgb / P3D_B[VEC_Z]) + cy_rgb;
    }

    offsetRgb = (unsigned int ) (( P2D_rgb_x*3 ) + ( P2D_rgb_y * rgbWidth * 3 ));
    undistortedRgbPtr =  undistortedRgb + offsetRgb;

   //we should take the color P2D_rgb_x , P2D_rgb_y
   *outputPTR = *undistortedRgbPtr;  ++outputPTR; ++undistortedRgbPtr;
   *outputPTR = *undistortedRgbPtr;  ++outputPTR; ++undistortedRgbPtr;
   *outputPTR = *undistortedRgbPtr;  ++outputPTR;

   ++x_d;
   //if (x_d==depthWidth) { x_d = 0; ++y_d; }
   y_d+= (x_d % rgbWidth != 0);
   x_d = x_d % rgbWidth;
  }

  return 1;
}

