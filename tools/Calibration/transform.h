#ifndef TRANSFORM_H_INCLUDED
#define TRANSFORM_H_INCLUDED

#include "calibration.h"


unsigned char *  registerUndistortedColorToUndistortedDepthFrame
                                          (
                                           unsigned char * undistortedRgb , unsigned int rgbWidth , unsigned int rgbHeight , struct calibration * rgbCalibration ,
                                           unsigned short * undistortedDepth , unsigned int depthWidth , unsigned int depthHeight , struct calibration * depthCalibration ,
                                           double * rotation3x3 , double * translation3x1 ,
                                           unsigned int * outputWidth , unsigned int * outputHeight
                                          );

#endif // TRANSFORM_H_INCLUDED
