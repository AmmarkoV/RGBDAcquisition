#include "bvh_interpolate.h"

#include <stdio.h>
#include <stdlib.h>


int bvh_InterpolateMotion(struct BVH_MotionCapture * mc)
{
  if (mc==0) { return 0; }

  float * newMotionValues = (float*) malloc(sizeof(float) * mc->motionValuesSize * 2 );

  if (newMotionValues!=0)
  {
   unsigned int i,z,target=0;
   for (i=0; i<mc->numberOfFrames-1; i++)
    {
      //First copy frame
      for (z=0; z<mc->numberOfValuesPerFrame; z++)
        { newMotionValues[target*mc->numberOfValuesPerFrame + z] = mc->motionValues[(i)*mc->numberOfValuesPerFrame + z]; }
      target++;
      //Then add an interpolated frame
      for (z=0; z<mc->numberOfValuesPerFrame; z++)
        { newMotionValues[target*mc->numberOfValuesPerFrame + z] = ( mc->motionValues[i*mc->numberOfValuesPerFrame + z] + mc->motionValues[(i+1)*mc->numberOfValuesPerFrame + z] ) / 2;  }
      target++;
    }

  //Copy last two frames
  i=mc->numberOfFrames-1;
  for (z=0; z<mc->numberOfValuesPerFrame; z++)
      { newMotionValues[target*mc->numberOfValuesPerFrame + z] = mc->motionValues[(i)*mc->numberOfValuesPerFrame + z]; }
   target++;
  for (z=0; z<mc->numberOfValuesPerFrame; z++)
      { newMotionValues[target*mc->numberOfValuesPerFrame + z] = mc->motionValues[(i)*mc->numberOfValuesPerFrame + z]; }


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
