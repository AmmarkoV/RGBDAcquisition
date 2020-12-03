#include "bvh_interpolate.h"

#include <stdio.h>
#include <stdlib.h>


int bvh_InterpolateMotion(struct BVH_MotionCapture * mc)
{
  if (mc==0) { return 0; }

  float * newMotionValues = (float*) malloc(sizeof(float) * mc->motionValuesSize * 2 );

  if (newMotionValues!=0)
  {
   unsigned int i,mID,target=0;
   for (i=0; i<mc->numberOfFrames-1; i++)
    {
      //First copy frame
      for (mID=0; mID<mc->numberOfValuesPerFrame; mID++)
        { newMotionValues[(target*mc->numberOfValuesPerFrame) + mID] = mc->motionValues[(i*mc->numberOfValuesPerFrame) + mID]; }
      target++;
      //Then add an interpolated frame
      for (mID=0; mID<mc->numberOfValuesPerFrame; mID++)
        { newMotionValues[(target*mc->numberOfValuesPerFrame) + mID] = ( mc->motionValues[(i*mc->numberOfValuesPerFrame) + mID] + mc->motionValues[ ((i+1)*mc->numberOfValuesPerFrame) + mID] ) / 2;  }
      target++;
    }

  //Copy last two frames
  i=mc->numberOfFrames-1;
  for (mID=0; mID<mc->numberOfValuesPerFrame; mID++)
      { newMotionValues[(target*mc->numberOfValuesPerFrame) + mID] = mc->motionValues[(i*mc->numberOfValuesPerFrame) + mID]; }
   target++;
  for (mID=0; mID<mc->numberOfValuesPerFrame; mID++)
      { newMotionValues[(target*mc->numberOfValuesPerFrame) + mID] = mc->motionValues[(i*mc->numberOfValuesPerFrame) + mID]; }


  float * oldMotionValues = mc->motionValues;
  mc->frameTime=mc->frameTime/2;
  mc->numberOfFrames=mc->numberOfFrames*2;
  mc->numberOfFramesEncountered=mc->numberOfFrames;
  mc->motionValuesSize=mc->motionValuesSize*2;
  mc->motionValues = newMotionValues;
  free(oldMotionValues);

  return 1;
  }

 return 0;
}
