#ifndef FACEDETECTION_H_INCLUDED
#define FACEDETECTION_H_INCLUDED


void InitFaceDetection(char * haarCascadePath , unsigned int x,unsigned int y);
void CloseFaceDetection() ;
unsigned int DetectFaces(unsigned char * colorPixels);

#endif // NITE2_H_INCLUDED
