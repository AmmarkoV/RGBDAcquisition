#include <stdio.h>
#include "automaticPlaneSegmentation.h"

#define MEMPLACE1(x,y,width) ( y * ( width ) + x )
#define ResultNormals 0

struct normalArray
{
    float p[3];
};

int automaticPlaneSegmentation(unsigned short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf , struct calibration * calib)
{
    fprintf(stderr,"doing automaticPlaneSegmentation()\n");
    double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(calib);
    if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform plane segmentation\n"); return 0; }

    unsigned int x,y;
    float p1[3]; float p2[3]; float p3[3]; float pN[3];
    float bestNormal[3]={0.0 , 0.0 , 0.0 };
    float thisNormal[3]={0.0 , 0.0 , 0.0 };


    struct normalArray result[ResultNormals]={0};
    unsigned int resultScore[ResultNormals]={0};

    int i=0;
    for (i=0; i<ResultNormals; i++)
    {
         x=rand()%(width-1);  y=rand()%(height-1);
         if (source[MEMPLACE1(x,y,width)]!=0)
             { transform2DProjectedPointTo3DPoint(calib , x, y , source[MEMPLACE1(x,y,width)] , &p1[0] , &p1[1] ,  &p1[2]); }

         x=rand()%(width-1);  y=rand()%(height-1);
         if (source[MEMPLACE1(x,y,width)]!=0)
             { transform2DProjectedPointTo3DPoint(calib , x, y , source[MEMPLACE1(x,y,width)] , &p2[0] , &p2[1] ,  &p2[2]); }

         x=rand()%(width-1);  y=rand()%(height-1);
         if (source[MEMPLACE1(x,y,width)]!=0)
             { transform2DProjectedPointTo3DPoint(calib , x, y , source[MEMPLACE1(x,y,width)] , &p3[0] , &p3[1] ,  &p3[2]); }


         crossProductFrom3Points( p1 , p2  , p3  , thisNormal);

         result[i].p[0]=thisNormal[0];
         result[i].p[1]=thisNormal[1];
         result[i].p[2]=thisNormal[2];

    }



    int z=0;
    for (i=0; i<ResultNormals; i++)
    {
      for (z=0; z<ResultNormals; z++)
      {
          if (z!=i)
          {
             resultScore[i]+=angleOfNormals(result[i].p,result[z].p);
          }
      }
    }

    float bestScore = 121230.0;
    for (i=0; i<ResultNormals; i++)
    {
      if (resultScore[i]<bestScore)
      {
        bestNormal[0]=result[i].p[0]; bestNormal[1]=result[i].p[1]; bestNormal[2]=result[i].p[2];
        bestScore = resultScore[i];
      }
    }


    free4x4Matrix(&m);
  return 0;
}
