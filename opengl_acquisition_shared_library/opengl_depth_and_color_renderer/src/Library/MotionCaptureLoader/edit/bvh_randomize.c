#include "bvh_randomize.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */


float randomFloatA( float minVal, float maxVal )
{
    if (maxVal<minVal)
    {
      float buf = minVal;
      minVal = maxVal;
      maxVal = buf;
    }

    float magnitude=maxVal-minVal;
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */
    float absoluteRandom = scale * magnitude;      /* [min, max] */
    float value = maxVal-absoluteRandom;
    //float value = minVal+absoluteRandom; <- same thing as above

    if (value<minVal) { fprintf(stderr,"randomFloat(%0.2f,%0.2f)=>%0.2f TOO SMALL\n",minVal,maxVal,value); } else
    if (value>maxVal) { fprintf(stderr,"randomFloat(%0.2f,%0.2f)=>%0.2f TOO BIG\n",minVal,maxVal,value); }

    return value;
}

int bvh_PerturbJointAnglesRange(
                           struct BVH_MotionCapture * mc,
                           unsigned int numberOfValues,
                           float  start,
                           float  end,
                           unsigned int specificChannel,
                           const char **argv,
                           unsigned int iplus2
                          )
{

  fprintf(stderr,"Asked to randomize %u Joint Angles in the range [%0.2f,%0.2f] deviation\n",numberOfValues,start,end);
  unsigned int * selectedJoints = (unsigned int *) malloc(sizeof(unsigned int) * mc->numberOfValuesPerFrame);
  if (selectedJoints!=0)
  {
    int success=1;

    memset(selectedJoints,0,sizeof(unsigned int)* mc->numberOfValuesPerFrame);
    BVHJointID jID=0;
    unsigned int mID=0;
    fprintf(stderr,"Randomizing : ");
    unsigned int i=0;
    for (i=iplus2+1; i<=iplus2+numberOfValues; i++)
     {
      fprintf(stderr,GREEN "%s " NORMAL,argv[i]);
      if (
           bvh_getJointIDFromJointName(
                                       mc,
                                       argv[i],
                                       &jID
                                      )
         )
         {
           unsigned int channelsEncountered=0;
           for (mID=0; mID<mc->numberOfValuesPerFrame; mID++)
           {
               if ( mc->motionToJointLookup[mID].jointID == jID )
               {
                 ++channelsEncountered;
                 if (specificChannel)
                 {
                   if (specificChannel==channelsEncountered)
                   {
                    selectedJoints[mID]=1;
                    fprintf(stderr,"Specific(%u) ",mID);
                   }
                 }  else
                 {
                   selectedJoints[mID]=1;
                   fprintf(stderr,"%u ",mID);
                 }
               }
           }
         } else
         {
           fprintf(stderr,RED "%s(not found) " NORMAL,argv[i]);
           success=0;
         }
     }
    fprintf(stderr,"\n");

     unsigned int fID=0;
     for (fID=0; fID<mc->numberOfFrames; fID++)
      {
       unsigned int mIDStart=fID*mc->numberOfValuesPerFrame;
       unsigned int mIDEnd=mIDStart+mc->numberOfValuesPerFrame;

       for (mID=mIDStart; mID<mIDEnd; mID++)
         {
           if (selectedJoints[mID-mIDStart])
           {
             //fprintf(stderr,"Was %0.2f ",mc->motionValues[mID+jID]);
             mc->motionValues[mID]+=randomFloatA(start,end);
             //fprintf(stderr,"Is %0.2f ",mc->motionValues[mID+jID]);
           }
         }
      }


    free(selectedJoints);
    return success;
  }

  return 0;
}





int bvh_PerturbJointAngles(
                           struct BVH_MotionCapture * mc,
                           unsigned int numberOfValues,
                           float  deviation,
                           const char **argv,
                           unsigned int iplus2
                          )
{
  fprintf(stderr,"Asked to randomize %u Joint Angles using a %0.2f (+- %0.2f) deviation\n",numberOfValues,deviation,(float) deviation/2);
  return bvh_PerturbJointAnglesRange(
                                     mc,
                                     numberOfValues,
                                     (float) -1*deviation/2,
                                     (float) deviation/2,
                                     0,
                                     argv,
                                     iplus2
                                    );
}












int bvh_eraseJoints(
                    struct BVH_MotionCapture * mc,
                    unsigned int numberOfValues,
                    unsigned int includeEndSites,
                    char **argv,
                    unsigned int iplus1
                   )
{
  //---------------------
  fprintf(stderr,"Asked to erase %u Joint Angles\n",numberOfValues);
  unsigned int * selectedJoints = (unsigned int *) malloc(sizeof(unsigned int) * mc->numberOfValuesPerFrame);
  if (selectedJoints!=0)
  {
    unsigned int success=1;
    memset(selectedJoints,0,sizeof(unsigned int)* mc->numberOfValuesPerFrame);
    BVHJointID jID=0;
    unsigned int mID=0;
    fprintf(stderr,"Erasing : ");
    unsigned int i=0;
    for (i=iplus1+1; i<=iplus1+numberOfValues; i++)
     {
      if (
           bvh_getJointIDFromJointName(
                                       mc,
                                       argv[i],
                                       &jID
                                      )
         )
         {
           fprintf(stderr,GREEN "%s " NORMAL,argv[i]);
           mc->jointHierarchy[jID].erase2DCoordinates=1;

           for (mID=0; mID<mc->numberOfValuesPerFrame; mID++)
           {
               if (mc->motionToJointLookup[mID].jointID == jID)
               {
                selectedJoints[mID]=1;
                fprintf(stderr,"%u ",mID);
               }
           }
           //-------------------------------------------------

           if(includeEndSites)
           {
             BVHJointID jIDES=jID;
             if (bhv_jointGetEndSiteChild(mc,jID,&jIDES))
               {
                 mc->jointHierarchy[jIDES].erase2DCoordinates=1;
                 fprintf(stderr,GREEN "%s_EndSite " NORMAL,argv[i]);

                 for (mID=0; mID<mc->numberOfValuesPerFrame; mID++)
                   {
                      if (mc->motionToJointLookup[mID].jointID == jIDES)
                         {
                           selectedJoints[mID]=1;
                           fprintf(stderr,"%u ",mID);
                         }
                  }
               }
           }

         } else
         {
           fprintf(stderr,RED "%s(not found) " NORMAL,argv[i]);
           success=0;
         }
     }
    fprintf(stderr,"\n");

     unsigned int fID=0;
     for (fID=0; fID<mc->numberOfFrames; fID++)
      {
       unsigned int mIDStart=fID*mc->numberOfValuesPerFrame;
       unsigned int mIDEnd=mIDStart+mc->numberOfValuesPerFrame;
       for (mID=mIDStart; mID<mIDEnd; mID++)
         {
           if (selectedJoints[mID-mIDStart])
           {
             //fprintf(stderr,"Was %0.2f ",mc->motionValues[mID+jID]);
             mc->motionValues[mID]=0.0;
             //fprintf(stderr,"Is %0.2f ",mc->motionValues[mID+jID]);
           }
         }
      }


    free(selectedJoints);
    return success;
  }

  return 0;
}




int bvh_RandomizePositionsBasedOn3D(
                                     struct BVH_MotionCapture * mc,
                                     float * minimumPosition,
                                     float * maximumPosition
                                    )
{
  fprintf(stderr,"Randomizing Positions of %u frames based on 3D coordinates\n",mc->numberOfFrames);
  fprintf(stderr,"min(%0.2f,%0.2f,%0.2f)",minimumPosition[0],minimumPosition[1],minimumPosition[2]);
  fprintf(stderr,"max(%0.2f,%0.2f,%0.2f)",maximumPosition[0],maximumPosition[1],maximumPosition[2]);
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;
   mc->motionValues[mID+0]=randomFloatA(minimumPosition[0],maximumPosition[0]);
   mc->motionValues[mID+1]=randomFloatA(minimumPosition[1],maximumPosition[1]);
   mc->motionValues[mID+2]=randomFloatA(minimumPosition[2],maximumPosition[2]);
  }
 return 1;
}


int bvh_RandomizeRotationsBasedOn3D(
                                     struct BVH_MotionCapture * mc,
                                     float * minimumRotation,
                                     float * maximumRotation
                                    )
{
  fprintf(stderr,"Randomizing Rotations of %u frames based on 3D coordinates\n",mc->numberOfFrames);
  fprintf(stderr,"min(%0.2f,%0.2f,%0.2f)",minimumRotation[0],minimumRotation[1],minimumRotation[2]);
  fprintf(stderr,"max(%0.2f,%0.2f,%0.2f)",maximumRotation[0],maximumRotation[1],maximumRotation[2]);
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;
   mc->motionValues[mID+3]=randomFloatA(minimumRotation[0],maximumRotation[0]);
   mc->motionValues[mID+4]=randomFloatA(minimumRotation[1],maximumRotation[1]);
   mc->motionValues[mID+5]=randomFloatA(minimumRotation[2],maximumRotation[2]);
  }
 return 1;
}




int bvh_RandomizePositionRotation(
                                  struct BVH_MotionCapture * mc,
                                  float * minimumPosition,
                                  float * minimumRotation,
                                  float * maximumPosition,
                                  float * maximumRotation
                                 )
{
  return ( (bvh_RandomizePositionsBasedOn3D(mc,minimumPosition,maximumPosition)) &&
            (bvh_RandomizeRotationsBasedOn3D(mc,minimumPosition,maximumPosition)) );
}



int bvh_RandomizePositionRotation2Ranges(
                                         struct BVH_MotionCapture * mc,
                                         float * minimumPositionRangeA,
                                         float * minimumRotationRangeA,
                                         float * maximumPositionRangeA,
                                         float * maximumRotationRangeA,
                                         float * minimumPositionRangeB,
                                         float * minimumRotationRangeB,
                                         float * maximumPositionRangeB,
                                         float * maximumRotationRangeB
                                        )
{
  fprintf(stderr,"Randomizing %u frames at two ranges\n",mc->numberOfFrames);
  fprintf(stderr,"Range A\n");
  fprintf(stderr,"min(Pos[%0.2f,%0.2f,%0.2f],",minimumPositionRangeA[0],minimumPositionRangeA[1],minimumPositionRangeA[2]);
  fprintf(stderr,"Rot[%0.2f,%0.2f,%0.2f])\n",minimumRotationRangeA[0],minimumRotationRangeA[1],minimumRotationRangeA[2]);
  fprintf(stderr,"max(Pos[%0.2f,%0.2f,%0.2f],",maximumPositionRangeA[0],maximumPositionRangeA[1],maximumPositionRangeA[2]);
  fprintf(stderr,"Rot[%0.2f,%0.2f,%0.2f])\n",maximumRotationRangeA[0],maximumRotationRangeA[1],maximumRotationRangeA[2]);
  fprintf(stderr,"Range B\n");
  fprintf(stderr,"min(Pos[%0.2f,%0.2f,%0.2f],",minimumPositionRangeB[0],minimumPositionRangeB[1],minimumPositionRangeB[2]);
  fprintf(stderr,"Rot[%0.2f,%0.2f,%0.2f])\n",minimumRotationRangeB[0],minimumRotationRangeB[1],minimumRotationRangeB[2]);
  fprintf(stderr,"max(Pos[%0.2f,%0.2f,%0.2f],",maximumPositionRangeB[0],maximumPositionRangeB[1],maximumPositionRangeB[2]);
  fprintf(stderr,"Rot[%0.2f,%0.2f,%0.2f])\n",maximumRotationRangeB[0],maximumRotationRangeB[1],maximumRotationRangeB[2]);

  //fprintf(stderr,"Exiting\n");
  //exit(0);

  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;
   float whichHalf = rand() / (float) RAND_MAX; /* [0, 1.0] */

   if (whichHalf<0.5)
           {
              mc->motionValues[mID+0]=randomFloatA(minimumPositionRangeA[0],maximumPositionRangeA[0]);
              mc->motionValues[mID+1]=randomFloatA(minimumPositionRangeA[1],maximumPositionRangeA[1]);
              mc->motionValues[mID+2]=randomFloatA(minimumPositionRangeA[2],maximumPositionRangeA[2]);
              mc->motionValues[mID+3]=randomFloatA(minimumRotationRangeA[0],maximumRotationRangeA[0]);
              mc->motionValues[mID+4]=randomFloatA(minimumRotationRangeA[1],maximumRotationRangeA[1]);
              mc->motionValues[mID+5]=randomFloatA(minimumRotationRangeA[2],maximumRotationRangeA[2]);
           } else
           {
              mc->motionValues[mID+0]=randomFloatA(minimumPositionRangeB[0],maximumPositionRangeB[0]);
              mc->motionValues[mID+1]=randomFloatA(minimumPositionRangeB[1],maximumPositionRangeB[1]);
              mc->motionValues[mID+2]=randomFloatA(minimumPositionRangeB[2],maximumPositionRangeB[2]);
              mc->motionValues[mID+3]=randomFloatA(minimumRotationRangeB[0],maximumRotationRangeB[0]);
              mc->motionValues[mID+4]=randomFloatA(minimumRotationRangeB[1],maximumRotationRangeB[1]);
              mc->motionValues[mID+5]=randomFloatA(minimumRotationRangeB[2],maximumRotationRangeB[2]);
           }
  }
 return 1;
}



void transform2DFProjectedPointTo3DPoint(float fX,float fY,float cX,float cY,unsigned int width,unsigned int height,
                                        float x2D , float y2D  , float depthValue ,
                                        float * x3D , float * y3D)
{
 *x3D = (float) (x2D - cX) * (depthValue / fX);
 *y3D = (float) (y2D - cY) * (depthValue / fY);
}






int bvh_RandomizePositionFrom2D(
                                 struct BVH_MotionCapture * mc,
                                 float * minimumRotation,
                                 float * maximumRotation,
                                 float minimumDepth,float maximumDepth,
                                 float fX,float fY,float cX,float cY,unsigned int width,unsigned int height
                                )
{
  fprintf(stderr,"Randomizing %u frames  using 2D randomizations \n",mc->numberOfFrames);

  unsigned int borderX=width/7; //8
  unsigned int borderY=height/4;//5

  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;

   mc->motionValues[mID+2]=randomFloatA(minimumDepth,maximumDepth);
   unsigned int x2D = borderX+ rand()%(width-borderX*2);
   unsigned int y2D = borderY+ rand()%(height-borderY*2);
   transform2DFProjectedPointTo3DPoint(fX,fY,cX,cY,width,height,(float) x2D,(float) y2D,mc->motionValues[mID+2],&mc->motionValues[mID+0],&mc->motionValues[mID+1]);

   mc->motionValues[mID+3]=randomFloatA(minimumRotation[0],maximumRotation[0]);
   mc->motionValues[mID+4]=randomFloatA(minimumRotation[1],maximumRotation[1]);
   mc->motionValues[mID+5]=randomFloatA(minimumRotation[2],maximumRotation[2]);
  }
 return 1;
}


int bvh_RandomizePositionFrom2DRotation2Ranges(
                                               struct BVH_MotionCapture * mc,
                                               float * minimumRotationRangeA,
                                               float * maximumRotationRangeA,
                                               float * minimumRotationRangeB,
                                               float * maximumRotationRangeB,
                                               float minimumDepth,float maximumDepth,
                                               float fX,float fY,float cX,float cY,unsigned int width,unsigned int height
                                              )
{
  fprintf(stderr,"Randomizing %u frames at two ranges\n",mc->numberOfFrames);

  //Randomize Positions using same codepath as bvh_RandomizePositionFrom2D (we also overwrite rotations but will get randomized again anyway)
  bvh_RandomizePositionFrom2D(
                               mc,
                               minimumRotationRangeA,
                               maximumRotationRangeA,
                               minimumDepth,maximumDepth,
                               fX,fY,cX,cY,width,height
                              );

  //Just Randomize Rotations like bvh_RandomizePositionRotation2Ranges
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   unsigned int mID=fID*mc->numberOfValuesPerFrame;
   float whichHalf = rand() / (float) RAND_MAX; /* [0, 1.0] */

   if (whichHalf<0.5)
           {
              mc->motionValues[mID+3]=randomFloatA(minimumRotationRangeA[0],maximumRotationRangeA[0]);
              mc->motionValues[mID+4]=randomFloatA(minimumRotationRangeA[1],maximumRotationRangeA[1]);
              mc->motionValues[mID+5]=randomFloatA(minimumRotationRangeA[2],maximumRotationRangeA[2]);
           } else
           {
              mc->motionValues[mID+3]=randomFloatA(minimumRotationRangeB[0],maximumRotationRangeB[0]);
              mc->motionValues[mID+4]=randomFloatA(minimumRotationRangeB[1],maximumRotationRangeB[1]);
              mc->motionValues[mID+5]=randomFloatA(minimumRotationRangeB[2],maximumRotationRangeB[2]);
           }
  }
 return 1;
}





int bvh_TestRandomizationLimitsXYZ(
                                   struct BVH_MotionCapture * mc,
                                   float * minimumPosition,
                                   float * maximumPosition
                                  )
{
  if (mc->numberOfFrames<8)
  {
    return 0;
  }
  mc->numberOfFrames=8;

  unsigned int fID,mID;
  //----------------------------------------
  fID=0; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=minimumPosition[0];  mc->motionValues[mID+1]=minimumPosition[1]; mc->motionValues[mID+2]=minimumPosition[2]; //Minimum X , Minimum Y , Minimum Z
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=minimumPosition[0]; mc->motionValues[mID+1]=minimumPosition[1];  mc->motionValues[mID+2]=maximumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=minimumPosition[0]; mc->motionValues[mID+1]=maximumPosition[1];  mc->motionValues[mID+2]=minimumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=minimumPosition[0]; mc->motionValues[mID+1]=maximumPosition[1];  mc->motionValues[mID+2]=maximumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=maximumPosition[0]; mc->motionValues[mID+1]=minimumPosition[1];  mc->motionValues[mID+2]=minimumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=maximumPosition[0]; mc->motionValues[mID+1]=minimumPosition[1];  mc->motionValues[mID+2]=maximumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=maximumPosition[0]; mc->motionValues[mID+1]=maximumPosition[1];  mc->motionValues[mID+2]=minimumPosition[2];
  //----------------------------------------
  ++fID; mID=fID*mc->numberOfValuesPerFrame;
  mc->motionValues[mID+0]=maximumPosition[0]; mc->motionValues[mID+1]=maximumPosition[1];  mc->motionValues[mID+2]=maximumPosition[2]; //Maximum X , Maximum Y, Maximum Z


  return 1;
}







