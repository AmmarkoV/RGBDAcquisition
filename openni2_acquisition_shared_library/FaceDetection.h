#ifndef FACEDETECTION_H_INCLUDED
#define FACEDETECTION_H_INCLUDED


int InitFaceDetection(char * haarCascadePath , unsigned int width ,unsigned int height);
int CloseFaceDetection() ;
unsigned int DetectFaces(unsigned int frameNumber , unsigned char * colorPixels , unsigned short * depthPixels, unsigned int maxHeadSize,unsigned int minHeadSize);

#endif // NITE2_H_INCLUDED
