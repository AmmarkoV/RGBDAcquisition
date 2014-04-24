#include <stdio.h>
#include "automaticPlaneSegmentation.h"

#define MEMPLACE1(x,y,width) ( y * ( width ) + x )
#define ResultNormals 30

struct normalArray
{

    float p1[3];
     unsigned int p1X , p1Y;

    float p2[3];
     unsigned int p2X , p2Y;

    float p3[3];
     unsigned int p3X , p3Y;

    float normal[3];
};

int ensureClockwise(unsigned int id , struct normalArray * result)
{
  return 0;

  int swapPos[3]={0,1,2 };


  if ( (result[id].p1X <=  result[id].p2X)  && (result[id].p2X <=  result[id].p3X)   )
  {
    //ok
  } else
  if ( (result[id].p1X >= result[id].p2X)  && (result[id].p2X <= result[id].p3X)   )
  {
    //ok
  }

  if ( (result[id].p1Y <  result[id].p2Y) && (result[id].p2Y <  result[id].p3Y) )
  {
    //ok

  }


}



int automaticPlaneSegmentation(unsigned short * source , unsigned int width , unsigned int height , struct SegmentationFeaturesDepth * segConf , struct calibration * calib)
{
    fprintf(stderr,"doing automaticPlaneSegmentation() VER\n");
    //double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(calib);
    //if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform plane segmentation\n"); return 0; }

    unsigned int x,y,depth;
    unsigned int bestNormal = 0;

    if (ResultNormals==0) { fprintf(stderr,"No Normals allowed cannot do automatic plane segmentation \n"); return 0; }

    struct normalArray result[ResultNormals]={0};
    unsigned int resultScore[ResultNormals]={0};

    unsigned int tries=0;
    int i=0;
    for (i=0; i<ResultNormals; i++)
    {
        fprintf(stderr,"TryNumber %u \n",i);
         result[i].p1[2]=0;
         tries=0; depth=0;
         while ( ( (depth!=0) || (tries==0) || (result[i].p1[2]==0) ) && (tries<10000) )
         {
          ++tries;
          x=rand()%(width-1);  y=rand()%(height-1); depth=source[MEMPLACE1(x,y,width)];
          if (depth!=0)  {
                           transform2DProjectedPointTo3DPoint(calib , x, y , depth , &result[i].p1[0] , &result[i].p1[1] ,  &result[i].p1[2]);
                           result[i].p1X=x; result[i].p1Y=y;
                         }
         }

         fprintf(stderr,"Point1(%u,%u) picked with depth %u \n",x,y,depth);

         result[i].p2[2]=0;
         tries=0; depth=0;
         while ( ( (depth!=0) || (tries==0) || (result[i].p2[2]==0) ) && (tries<10000) )
         {
          ++tries;
          x=rand()%(width-1);  y=rand()%(height-1); depth=source[MEMPLACE1(x,y,width)];
          if (depth!=0) {
                          transform2DProjectedPointTo3DPoint(calib , x, y , depth , &result[i].p2[0] , &result[i].p2[1] ,  &result[i].p2[2]);
                          result[i].p2X=x; result[i].p2Y=y;
                        }
         }

         fprintf(stderr,"Point2(%u,%u) picked with depth %u \n",x,y,depth);

         result[i].p3[2]=0;
         tries=0; depth=0;
         while ( ( (depth!=0) || (tries==0) || (result[i].p3[2]==0) ) && (tries<10000) )
         {
          ++tries;
          x=rand()%(width-1);  y=rand()%(height-1); depth=source[MEMPLACE1(x,y,width)];
          if (depth!=0) {
                          transform2DProjectedPointTo3DPoint(calib , x, y , depth , &result[i].p3[0] , &result[i].p3[1] ,  &result[i].p3[2]);
                          result[i].p3X=x; result[i].p3Y=y;
                        }
         }

         fprintf(stderr,"Point3(%u,%u) picked with depth %u \n",x,y,depth);

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

    for (i=0; i<ResultNormals; i++)
    {
        ensureClockwise(i , result);
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

   fprintf(stderr,"Best Points are \n %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n " ,
                         result[bestNormal].p1[0] ,  result[bestNormal].p1[1] ,  result[bestNormal].p1[2] ,
                         result[bestNormal].p2[0] ,  result[bestNormal].p2[1] ,  result[bestNormal].p2[2] ,
                         result[bestNormal].p3[0] ,  result[bestNormal].p3[1] ,  result[bestNormal].p3[2]
         );

   fprintf(stderr,"AUTOMATIC SHUTDOWN OF SEGMENTATION SO THAT DOES NOT DESTORY OUTPUT\n");
   segConf->autoPlaneSegmentation=0;

   // free4x4Matrix(&m);
  return 1;
}
