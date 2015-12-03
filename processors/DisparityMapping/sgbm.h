#ifndef SGBM_H_INCLUDED
#define SGBM_H_INCLUDED

int newKindOfDisplayCalibrationReading(char * disparityCalibrationPath);
int doSGBM(unsigned char * colorFrame , unsigned int colorWidth ,unsigned int colorHeight ,  unsigned int swapColorFeeds
           , unsigned int SADWindowSize , unsigned int shiftYLeft , unsigned int shiftYRight ,
           unsigned int speckleRange , char * disparityCalibrationPath);

#endif // SGBM_H_INCLUDED
