#ifndef STEREO_CALIBRATE_H_INCLUDED
#define STEREO_CALIBRATE_H_INCLUDED

int doCalibrationStep(char* imageLeftRGB,char* imageRightRGB,unsigned int imageRGBWidth,unsigned int imageRGBHeight,unsigned int horizontalSquares,unsigned int verticalSquares,float calibSquareSize);

#endif // STEREO_CALIBRATE_H_INCLUDED
