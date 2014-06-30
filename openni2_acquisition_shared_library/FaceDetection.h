#ifndef FACEDETECTION_H_INCLUDED
#define FACEDETECTION_H_INCLUDED


#include "../tools/Calibration/calibration.h"


struct detectedFace
{
  unsigned int observationNumber , observationTotal;

  unsigned int sX , sY , tileWidth , tileHeight , distance;
  float headX, headY, headZ;
};


//Leverage Depth Information using fast min/max head sizes heuristic ( this is great , should be always on for RGBD input)
extern int useDepthHeadMinMaxSizeHeuristic;

//Leverage Skin Color heuristic , this doesnt work very good for people of different skin color
extern int useHistogramHeuristic;



int InitFaceDetection(char * haarCascadePath);
int CloseFaceDetection() ;
int registerFaceDetectedEvent(void * callback);





unsigned int DetectFaces(unsigned int frameNumber ,
                         unsigned char * colorPixels ,  unsigned int colorWidth ,unsigned int colorHeight ,
                         unsigned short * depthPixels ,   unsigned int depthWidth ,unsigned int depthHeight ,
                         struct calibration * calib ,
                         unsigned int maxHeadSize,unsigned int minHeadSize);
#endif // NITE2_H_INCLUDED
