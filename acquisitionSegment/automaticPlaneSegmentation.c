#include <stdio.h>
#include "automaticPlaneSegmentation.h"



int automaticPlaneSegmentation(unsigned short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf , struct calibration * calib )
{
    double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(calib);
    if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform plane segmentation\n"); return 0; }

    unsigned int x,y;
    float p1[3]; float p2[3]; float p3[3]; float pN[3];
    float normal[3]={0.0 , 0.0 , 0.0 };

    crossProduct( p1 , p2  , p3  , normal);


    int i=0;
    for (i=0; i<20; i++)
    {
         x=rand()%(width-1);
         y=rand()%(height-1);

         transform2DProjectedPointTo3DPoint(calib , x, y , source[x*width+y] , &p1[0] , &p1[1] ,  &p1[2]);
    }

    free4x4Matrix(&m);
  return 0;
}
