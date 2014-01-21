#ifndef FACEDETECTION_H_INCLUDED
#define FACEDETECTION_H_INCLUDED


void InitFaceDetection(char * haarCascadePath , unsigned int width ,unsigned int height);
void CloseFaceDetection() ;
unsigned int DetectFaces(unsigned char * colorPixels , unsigned int maxHeadSize,unsigned int minHeadSize);

#endif // NITE2_H_INCLUDED
