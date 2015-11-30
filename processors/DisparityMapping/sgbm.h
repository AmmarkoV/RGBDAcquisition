#ifndef SGBM_H_INCLUDED
#define SGBM_H_INCLUDED

int newKindOfDisplayCalibrationReading(char * disparityCalibrationPath);
int doSGBM(unsigned char * colorFrame , unsigned int colorWidth , unsigned int colorHeight , unsigned int swapColorFeeds , unsigned int SADWindowSize , char * disparityCalibrationPath);

#endif // SGBM_H_INCLUDED
