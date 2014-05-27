#include <stdio.h>
#include <stdlib.h>
#include "automaticPlaneSegmentation.h"
#include "imageProcessing.h"
#include "../tools/Quasirandomness/quasirandomness.h"

#define MEMPLACE1(x,y,width) ( y * ( width ) + x )
#define GET_RANDOM_DIM(width,bounds) (bounds+rand()%(width-1-bounds))

unsigned int minimumAcceptedDepths = 830;
unsigned int maximumAcceptedDepths = 3000;

#define NeighborhoodNormalCombos 6
#define ResultNormals 64
#define MaxTriesPerPoint 1000


unsigned int neighborhoodHalfWidth = 5;
unsigned int neighborhoodHalfHeight = 5;
enum NEIGHBORHOOD_OF_POINTS
{
   NH_TOPLEFT = 0 ,
   NH_TOPRIGHT ,
   NH_MIDLEFT,
   NH_CENTER,
   NH_MIDRIGHT,
   NH_BOTLEFT,
   NH_BOTRIGHT
};

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

    struct TriplePoint point;
    float normal[3];
};


inline float getDepthValue(unsigned short * source , unsigned int x, unsigned int y , unsigned int width)
{
  return source[MEMPLACE1(x,y,width)];
}

unsigned int decideNormalAround3DPoint(unsigned short * source , struct calibration * calib , unsigned int x , unsigned int y  , unsigned int width , unsigned int height , float normal[3] )
{

   //---------------------------------------------------
   //        *0                                   *1
   //
   //        *2        *3 Central Point *         *4
   //
   //        *5                                   *6
   //---------------------------------------------------
   //        Calculated normals ( triangles will be )
   //        0 , 2 , 3 -> Normal 0
   //        0 , 3 , 1 -> Normal 1
   //        1 , 3 , 4 -> Normal 2
   //        2 , 5 , 3 -> Normal 3
   //        3 , 6 , 4 -> Normal 4
   //        3 , 5 , 6 -> Normal 5
   //---------------------------------------------------
   unsigned int normalSeries[NeighborhoodNormalCombos][3] = { { 0, 2, 3 }, { 0, 3, 1 }, { 1, 3, 4 } , { 2, 5, 3 } , { 3, 6, 4 } , { 3, 5, 6 } };

   struct TriplePoint neighbors[7]={0};
   struct normalArray neighborNormals[NeighborhoodNormalCombos]={0};
   unsigned int normalOK[NeighborhoodNormalCombos]={0};


   //We populate the neighborhood with values Z coordinate will signal existing/inexisting points
   neighbors[NH_CENTER].coord[X]=x;
   neighbors[NH_CENTER].coord[Y]=y;
   neighbors[NH_CENTER].coord[Z]=getDepthValue(source,neighbors[NH_CENTER].coord[X],neighbors[NH_CENTER].coord[Y],width);

   if ( (x>neighborhoodHalfWidth) && (y>neighborhoodHalfHeight) )
       { neighbors[NH_TOPLEFT].coord[X]=x-neighborhoodHalfWidth; neighbors[NH_TOPLEFT].coord[Y]=y-neighborhoodHalfHeight;
         neighbors[NH_TOPLEFT].coord[Z]=getDepthValue(source,neighbors[NH_TOPLEFT].coord[X],neighbors[NH_TOPLEFT].coord[Y],width); }

   if ( (x+neighborhoodHalfWidth>width) && (y>neighborhoodHalfHeight) )
      { neighbors[NH_TOPRIGHT].coord[X]=x+neighborhoodHalfWidth; neighbors[NH_TOPRIGHT].coord[Y]=y-neighborhoodHalfHeight;
        neighbors[NH_TOPRIGHT].coord[Z]=getDepthValue(source,neighbors[NH_TOPRIGHT].coord[X],neighbors[NH_TOPRIGHT].coord[Y],width); }

   if ( (x>neighborhoodHalfWidth) && (y>neighborhoodHalfHeight) )
      { neighbors[NH_MIDLEFT].coord[X]=x-neighborhoodHalfWidth; neighbors[NH_MIDLEFT].coord[Y]=y;
        neighbors[NH_MIDLEFT].coord[Z]=getDepthValue(source,neighbors[NH_MIDLEFT].coord[X],neighbors[NH_MIDLEFT].coord[Y],width); }

   if ( (x+neighborhoodHalfWidth>width) && (y>neighborhoodHalfHeight) )
      { neighbors[NH_MIDRIGHT].coord[X]=x+neighborhoodHalfWidth; neighbors[NH_MIDRIGHT].coord[Y]=y;
        neighbors[NH_MIDRIGHT].coord[Z]=getDepthValue(source,neighbors[NH_MIDRIGHT].coord[X],neighbors[NH_MIDRIGHT].coord[Y],width); }

   if ( (x>neighborhoodHalfWidth) && (y+neighborhoodHalfHeight>height) )
       { neighbors[NH_BOTLEFT].coord[X]=x-neighborhoodHalfWidth; neighbors[NH_BOTLEFT].coord[Y]=y+neighborhoodHalfHeight;
         neighbors[NH_BOTLEFT].coord[Z]=getDepthValue(source,neighbors[NH_BOTLEFT].coord[X],neighbors[NH_BOTLEFT].coord[Y],width); }

   if ( (x+neighborhoodHalfWidth>width) && (y+neighborhoodHalfHeight>height) )
      { neighbors[NH_BOTRIGHT].coord[X]=x+neighborhoodHalfWidth; neighbors[NH_BOTRIGHT].coord[Y]=y+neighborhoodHalfHeight;
        neighbors[NH_BOTRIGHT].coord[Z]=getDepthValue(source,neighbors[NH_BOTRIGHT].coord[X],neighbors[NH_BOTRIGHT].coord[Y],width); }

   unsigned int i=0;


   //Project Points to get real 3D coordinates
   unsigned int x2D,y2D,d3D;
   for (i=0; i<7; i++)
   {
        x2D = (unsigned int) neighbors[i].coord[X];
        y2D = (unsigned int) neighbors[i].coord[Y];
        d3D = (unsigned int) neighbors[i].coord[Z];

       transform2DProjectedPointTo3DPoint( calib , x2D , y2D  , d3D ,
                                           &neighbors[i].coord[X] ,
                                           &neighbors[i].coord[Y] ,
                                           &neighbors[i].coord[Z]
                                          );
   }



   //We have an array of normals ready , lets populate the normals now
   for (i=0; i<NeighborhoodNormalCombos; i++)
   {
      if ( //If all three points have a depth
           (neighbors[normalSeries[i][0]].coord[Z]!=0) &&
           (neighbors[normalSeries[i][1]].coord[Z]!=0) &&
           (neighbors[normalSeries[i][2]].coord[Z]!=0)
         )
          { //Find the normal
            crossProductFrom3Points( neighbors[normalSeries[i][0]].coord ,
                                     neighbors[normalSeries[i][1]].coord ,
                                     neighbors[normalSeries[i][2]].coord ,
                                     neighborNormals[i].normal);
            //Mark the normal as found
            normalOK[i]=1;
          }
   }

   unsigned int samples = 0;
   normal[0]=0; normal[1]=0; normal[2]=0;
   for (i=0; i<NeighborhoodNormalCombos; i++)
   {
     if (normalOK[i])
     {
       ++samples;
       normal[0]+=neighborNormals[i].normal[0];
       normal[1]+=neighborNormals[i].normal[1];
       normal[2]+=neighborNormals[i].normal[2];
     }
   }

   if (samples>0)
   {
     normal[0]/=samples;
     normal[1]/=samples;
     normal[2]/=samples;

     return 1;
   }

   return 0;
}





int automaticPlaneSegmentation(unsigned short * source , unsigned int width , unsigned int height , float offset, struct SegmentationFeaturesDepth * segConf , struct calibration * calib)
{
    fprintf(stderr,"doing automaticPlaneSegmentation( using RANSAC with %u points )\n",ResultNormals);

    unsigned int boundDistance=10;
    unsigned int x,y,depth;
    unsigned int bestNormal = 0;

    if (ResultNormals==0) { fprintf(stderr,"No Normals allowed cannot do automatic plane segmentation \n"); return 0; }

    struct normalArray result[ResultNormals]={0};
    unsigned int resultScore[ResultNormals]={0};


    struct TriplePoint legend;
    legend.coord[X]=0.016560; legend.coord[Y]=-0.826509; legend.coord[Z]=-0.562679;

    struct quasiRandomizerContext qrc;
    initializeQuasirandomnessContext(&qrc,width,height,0);
    float rX,rY,rZ;

    unsigned int tries=0;
    int i=0;
    for (i=0; i<ResultNormals; i++)
    {
        tries=0; depth=0;
         while ( ( (depth==0) || (tries==0) || (result[i].point.coord[Z]==0) ) && (tries<MaxTriesPerPoint) )
         {
          ++tries;
          getNextRandomPoint(&qrc,&rX,&rY,&rZ);
          //x=GET_RANDOM_DIM(width,boundDistance);  y=GET_RANDOM_DIM(height,boundDistance);
          x=(unsigned int) rX;
          y=(unsigned int) rY;
          if ( (x<width) && (y<height) )
          {
           depth=source[MEMPLACE1(x,y,width)];
           if ( (minimumAcceptedDepths<depth) && (depth<maximumAcceptedDepths) )
                         {
                           decideNormalAround3DPoint(source , calib , x , y  , width , height , result[i].normal );
                         } else
                         {
                           depth=0; //We will not use this point , please find another one
                         }
          } else
          {
            depth=0;
          }
         }
         fprintf(stderr,"Normal %u(%u,%u) picked with depth %u , after %u tries \n",i,x,y,depth,tries);
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


    float MAX_SCORE = 121230.0;
    float bestScore = MAX_SCORE;
    for (i=0; i<ResultNormals; i++)
    {
      if (resultScore[i]<bestScore)
      {
        bestNormal = i;
        bestScore = resultScore[i];
      }
    }

   if (bestScore == MAX_SCORE)
   {
     fprintf(stderr,"Failed to get an automatic plane :( \n");
     return 0;
   }

   segConf->enablePlaneSegmentation=1;
   segConf->planeNormalOffset=offset; //<- this is to ensure a good auto segmentation
   segConf->doNotGenerateNormalFrom3Points=1; // <- we have a normal and a point , we dont have 3 points

   segConf->center[0] = result[bestNormal].point.coord[0];
   segConf->center[1] = result[bestNormal].point.coord[1];
   segConf->center[2] = result[bestNormal].point.coord[2];

   segConf->normal[0] = result[bestNormal].normal[0];
   segConf->normal[1] = result[bestNormal].normal[1];
   segConf->normal[2] = result[bestNormal].normal[2];
   fprintf(stderr,"Picked result %u with score %0.2f \n",bestNormal , bestScore);



   fprintf(stderr,"Best Points are \n point %0.2f %0.2f %0.2f \n normal %0.2f %0.2f %0.2f \n" ,
                         result[bestNormal].point.coord[0] ,  result[bestNormal].point.coord[1] ,  result[bestNormal].point.coord[2] ,
                         result[bestNormal].normal[0] ,  result[bestNormal].normal[1] ,  result[bestNormal].normal[2]
         );

   fprintf(stderr,"Automatic shutdown of automatic plane segmentation so it does not feed on itself on the next frame\n");
   segConf->autoPlaneSegmentation=0;
  return 1;
}
