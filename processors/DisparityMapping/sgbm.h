#ifndef SGBM_H_INCLUDED
#define SGBM_H_INCLUDED



#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "cvaux.h"

int oldKindOfDisplayCalibrationReading(char * disparityCalibrationPath);
int newKindOfDisplayCalibrationReading(char * disparityCalibrationPath);

int doSGBM( cv::Mat *leftBGR,cv::Mat *rightBGR , unsigned int SADWindowSize ,  unsigned int speckleRange, char * disparityCalibrationPath);

#endif // SGBM_H_INCLUDED
