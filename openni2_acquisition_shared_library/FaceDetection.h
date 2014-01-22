#ifndef FACEDETECTION_H_INCLUDED
#define FACEDETECTION_H_INCLUDED


#include "../tools/Calibration/calibration.h"
int InitFaceDetection(char * haarCascadePath , unsigned int width ,unsigned int height);
int CloseFaceDetection() ;
int registerFaceDetectedEvent(void * callback);
unsigned int DetectFaces(unsigned int frameNumber , unsigned char * colorPixels , unsigned short * depthPixels, struct calibration * calib ,unsigned int maxHeadSize,unsigned int minHeadSize);

#endif // NITE2_H_INCLUDED
