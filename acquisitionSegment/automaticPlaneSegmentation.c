#include <stdio.h>
#include <stdlib.h>
#include "automaticPlaneSegmentation.h"
#include "imageProcessing.h"

#define MEMPLACE1(x,y,width) ( y * ( width ) + x )
#define ResultNormals 256
#define MaxTriesPerPoint 1000


enum pShorthand
{
  X=0,Y,Z
};



struct TriplePoint
{
 float coord[3];
};

struct normalArray
{

    struct TriplePoint point[3];

    float normal[3];
};


int swapResultPoints(unsigned int id ,struct normalArray * result, unsigned int swapIDA,unsigned int swapIDB)
{
  if (swapIDA==swapIDB) { return 1; }

  float tmp=0.0;
  unsigned int i=0;
  for (i=0; i<3; i++)
     {
        tmp = result[id].point[swapIDA].coord[i];
        result[id].point[swapIDA].coord[i] = result[id].point[swapIDB].coord[i];
        result[id].point[swapIDB].coord[i] = tmp;
     }
  return 1;
}


int ensureClockwise(unsigned int id , struct normalArray * result)
{
  unsigned int swapA=0,swapB=1;

  struct TriplePoint legend;
  legend.coord[X]=0.016560; legend.coord[Y]=-0.826509; legend.coord[Z]=-0.562679;


  float retres = dotProduct(result->normal , legend.coord );

  if (retres<0.0) { swapResultPoints(id,result,swapA,swapB); }

  return 1;
}



int automaticPlaneSegmentation(unsigned short * source , unsigned int width , unsigned int height , float offset, struct SegmentationFeaturesDepth * segConf , struct calibration * calib)
{
    fprintf(stderr,"doing automaticPlaneSegmentation( using RANSAC with %u points ) VER\n",ResultNormals);
    //double * m = allocate4x4MatrixForPointTransformationBasedOnCalibration(calib);
    //if (m==0) {fprintf(stderr,"Could not allocate a 4x4 matrix , cannot perform plane segmentation\n"); return 0; }
    unsigned int boundDistance=10;
    unsigned int x,y,depth;
    unsigned int bestNormal = 0;

    if (ResultNormals==0) { fprintf(stderr,"No Normals allowed cannot do automatic plane segmentation \n"); return 0; }

    struct normalArray result[ResultNormals]={0};
    unsigned int resultScore[ResultNormals]={0};


    struct TriplePoint legend;
    legend.coord[X]=0.016560; legend.coord[Y]=-0.826509; legend.coord[Z]=-0.562679;

    unsigned int tries=0;
    int i=0;
    for (i=0; i<ResultNormals; i++)
    {
        fprintf(stderr,"TryNumber %u \n",i);
         result[i].point[0].coord[Z]=0;
         tries=0; depth=0;
         while ( ( (depth==0) || (tries==0) || (result[i].point[0].coord[Z]==0) ) && (tries<MaxTriesPerPoint) )
         {
          ++tries;
          x=boundDistance+rand()%(width-1-boundDistance);  y=boundDistance+rand()%(height-1-boundDistance); depth=source[MEMPLACE1(x,y,width)];
          if (depth!=0)  {
                           transform2DProjectedPointTo3DPoint(calib , x, y , depth , &result[i].point[0].coord[X] , &result[i].point[0].coord[Y] ,  &result[i].point[0].coord[Z]);
                         }
         }

         fprintf(stderr,"Point1(%u,%u) picked with depth %u , after %u tries \n",x,y,depth,tries);

         result[i].point[1].coord[Z]=0;
         tries=0; depth=0;
         while ( ( (depth==0) || (tries==0) || (result[i].point[1].coord[Z]==0) ) && (tries<MaxTriesPerPoint) )
         {
          ++tries;
          x=boundDistance+rand()%(width-1-boundDistance);  y=boundDistance+rand()%(height-1-boundDistance); depth=source[MEMPLACE1(x,y,width)];
          if (depth!=0) {
                          transform2DProjectedPointTo3DPoint(calib , x, y , depth , &result[i].point[1].coord[X] , &result[i].point[1].coord[Y] ,  &result[i].point[1].coord[Z]);
                        }
         }

         fprintf(stderr,"Point2(%u,%u) picked with depth %u , after %u tries \n",x,y,depth,tries);

         result[i].point[2].coord[Z]=0;
         tries=0; depth=0;
         while ( ( (depth==0) || (tries==0) || (result[i].point[2].coord[Z]==0) ) && (tries<MaxTriesPerPoint) )
         {
          ++tries;
          x=boundDistance+rand()%(width-1-boundDistance);  y=boundDistance+rand()%(height-1-boundDistance); depth=source[MEMPLACE1(x,y,width)];
          if (depth!=0) {
                          transform2DProjectedPointTo3DPoint(calib , x, y , depth , &result[i].point[2].coord[X] , &result[i].point[2].coord[Y] ,  &result[i].point[2].coord[Z]);
                        }
         }

         fprintf(stderr,"Point3(%u,%u) picked with depth %u , after %u tries \n",x,y,depth,tries);

         fprintf(stderr,"3 Points are %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n " ,
                         result[i].point[0].coord[X] ,  result[i].point[0].coord[Y] ,  result[i].point[0].coord[Z] ,
                         result[i].point[1].coord[X] ,  result[i].point[1].coord[Y] ,  result[i].point[1].coord[Z] ,
                         result[i].point[2].coord[X] ,  result[i].point[2].coord[Y] ,  result[i].point[2].coord[Z]
                );

         crossProductFrom3Points( result[i].point[0].coord , result[i].point[1].coord  , result[i].point[2].coord  , result[i].normal);

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

      //Adding angle penalty according to legend ( to try and match it
      resultScore[i]+=angleOfNormals(result[i].normal,legend.coord);
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
    segConf->planeNormalOffset=offset; //<- this is to ensure a good auto segmentation
    for (i=0; i<3; i++)
      {
       segConf->p1[i]=result[bestNormal].point[0].coord[i];
       segConf->p2[i]=result[bestNormal].point[1].coord[i];
       segConf->p3[i]=result[bestNormal].point[2].coord[i];
      }

   fprintf(stderr,"Best Points are \n %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n %0.2f %0.2f %0.2f \n " ,
                         result[bestNormal].point[0].coord[0] ,  result[bestNormal].point[0].coord[1] ,  result[bestNormal].point[0].coord[2] ,
                         result[bestNormal].point[1].coord[0] ,  result[bestNormal].point[1].coord[1] ,  result[bestNormal].point[1].coord[2] ,
                         result[bestNormal].point[2].coord[0] ,  result[bestNormal].point[2].coord[1] ,  result[bestNormal].point[2].coord[2]
         );

   fprintf(stderr,"AUTOMATIC SHUTDOWN OF SEGMENTATION SO THAT DOES NOT DESTORY OUTPUT\n");
   segConf->autoPlaneSegmentation=0;

   // free4x4Matrix(&m);
  return 1;
}
