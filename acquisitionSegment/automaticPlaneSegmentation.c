#include <stdio.h>
#include "automaticPlaneSegmentation.h"

#define MEMPLACE1(x,y,width) ( y * ( width ) + x )
#define ResultNormals 0

struct normalArray
{
    float p1[3];
    float p2[3];
    float p3[3];
    float normal[3];
};

int automaticPlaneSegmentation(unsigned short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf , struct calibration * calib)
{
    fprintf(stderr,"doing automaticPlaneSegmentation()\n");
    //double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(calib);
    //if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform plane segmentation\n"); return 0; }

    unsigned int x,y;
    unsigned int bestNormal = 0;

    struct normalArray result[ResultNormals]={0};
    unsigned int resultScore[ResultNormals]={0};

    unsigned int tries=0;
    int i=0;
    for (i=0; i<ResultNormals; i++)
    {

         tries=0;
         while ( ( (source[MEMPLACE1(x,y,width)]!=0) || (tries==0) ) && (tries<10000) )
         {
          ++tries;
          x=rand()%(width-1);  y=rand()%(height-1);
          if (source[MEMPLACE1(x,y,width)]!=0)
                 { transform2DProjectedPointTo3DPoint(calib , x, y , source[MEMPLACE1(x,y,width)] , &result[i].p1[0] , &result[i].p1[1] ,  &result[i].p1[2]); }
         }

         tries=0;
         while ( ( (source[MEMPLACE1(x,y,width)]!=0) || (tries==0) ) && (tries<10000) )
         {
          ++tries;
          x=rand()%(width-1);  y=rand()%(height-1);
          if (source[MEMPLACE1(x,y,width)]!=0)
                 { transform2DProjectedPointTo3DPoint(calib , x, y , source[MEMPLACE1(x,y,width)] , &result[i].p2[0] , &result[i].p2[1] ,  &result[i].p2[2]); }
         }


         tries=0;
         while ( ( (source[MEMPLACE1(x,y,width)]!=0) || (tries==0) ) && (tries<10000) )
         {
          ++tries;
          x=rand()%(width-1);  y=rand()%(height-1);
          if (source[MEMPLACE1(x,y,width)]!=0)
                 { transform2DProjectedPointTo3DPoint(calib , x, y , source[MEMPLACE1(x,y,width)] , &result[i].p3[0] , &result[i].p3[1] ,  &result[i].p3[2]); }
         }

         fprintf(stderr,"3 Points are %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n " ,
                         result[i].p1[0] ,  result[i].p1[1] ,  result[i].p1[2] ,
                         result[i].p2[0] ,  result[i].p2[1] ,  result[i].p2[2] ,
                         result[i].p3[0] ,  result[i].p3[1] ,  result[i].p3[2]
                );

         crossProductFrom3Points( result[i].p1 , result[i].p2  , result[i].p3  , result[i].normal);

    }



    int z=0;
    for (i=0; i<ResultNormals; i++)
    {
      for (z=0; z<ResultNormals; z++)
      {
          if (z!=i)
          {
             resultScore[i]+=angleOfNormals(result[i].normal,result[z].normal);
          }
      }
    }

    float bestScore = 121230.0;
    for (i=0; i<ResultNormals; i++)
    {
      if (resultScore[i]<bestScore)
      {
        bestNormal = i;
        bestScore = resultScore[i];
      }
    }


    fprintf(stderr,"Picked result %u with score %0.2f \n",bestNormal , bestScore);

    segConf->enablePlaneSegmentation=1;
    for (i=0; i<3; i++)
      {
       segConf->p1[i]=result[bestNormal].p1[i];
       segConf->p2[i]=result[bestNormal].p2[i];
       segConf->p3[i]=result[bestNormal].p3[i];
      }

   fprintf(stderr,"AUTOMATIC SHUTDOWN OF SEGMENTATION SO THAT DOES NOT DESTORY OUTPUT\n");
   segConf->autoPlaneSegmentation=0;

   // free4x4Matrix(&m);
  return 1;
}
