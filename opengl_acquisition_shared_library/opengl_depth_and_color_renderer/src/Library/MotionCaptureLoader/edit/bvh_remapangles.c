#include "bvh_remapangles.h"

#include <stdio.h>
#include <math.h>


/*
//THIS IS NOT USED ANYWHERE
float bvh_constrainAngleCentered180(float angle)
{
   angle = fmod(angle,360.0);
   if (angle<0.0)
     { angle+=360.0; }
   return angle;
}
*/


// We have circles A , B and C and we are trying to map circles A and B to circle C
// Because neural networks get confused when coordinates jump from 0 to 360
//
//                                -360 A 0
//
//                                   0 B 360
//
//                                -180 C 180
//
//
//      -270A . 90B . -90C             *                 90C  270B  -90A
//
//
//                                  -1 C 1
//
//                                 179 B 181
//
//                                -181 C -179
//
//
//We want to add 180 degrees to the model so 0 is oriented towards us..!
float bvh_constrainAngleCentered0(float angle,unsigned int flipOrientation)
{
    float angleFrom_minus360_to_plus360;
    float angleRotated = angle+180;

     if (angleRotated<0.0)
     {
       angleFrom_minus360_to_plus360 = (-1*fmod(-1*(angleRotated),360.0))+180;
     } else
     {
       angleFrom_minus360_to_plus360 = (fmod((angleRotated),360.0))-180;
     }

    //If we want to flip orientation we just add or subtract 180 depending on the case
    //To retrieve correct orientatiation we do the opposite
    if (flipOrientation)
    {
      if (angleFrom_minus360_to_plus360<0.0) { angleFrom_minus360_to_plus360+=180.0; } else
      if (angleFrom_minus360_to_plus360>0.0) { angleFrom_minus360_to_plus360-=180.0; } else
                                             { angleFrom_minus360_to_plus360=180.0;  }
    }

   return angleFrom_minus360_to_plus360;
}



float bvh_RemapAngleCentered0(float angle, unsigned int constrainOrientation)
{
    /*
   float angleShifted = angle;
   //We want to add 180 degrees to the model so 0 is oriented towards us..!
   switch (constrainOrientation)
   {
      case BVH_ENFORCE_NO_ORIENTATION :                          return bvh_constrainAngleCentered0(angleShifted,0); break;
      case BVH_ENFORCE_FRONT_ORIENTATION :                       return bvh_constrainAngleCentered0(angleShifted,0); break;
      case BVH_ENFORCE_BACK_ORIENTATION :                        return bvh_constrainAngleCentered0(angleShifted,1); break;
      case BVH_ENFORCE_LEFT_ORIENTATION :    angleShifted+=90.0; return bvh_constrainAngleCentered0(angleShifted,0); break;
      case BVH_ENFORCE_RIGHT_ORIENTATION :   angleShifted+=90.0; return bvh_constrainAngleCentered0(angleShifted,1); break;
   };
*/
   fprintf(stderr,"bvh_RemapAngleCentered0: Did not change angle using deprecated code..\n");
  return angle;
}



int bvh_swapJointRotationAxis(struct BVH_MotionCapture * bvh,char inputRotationOrder,char swappedRotationOrder)
{
  if ( (bvh!=0) && (bvh->jointHierarchy!=0) )
  {
    for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
    {
        if (bvh->jointHierarchy[jID].channelRotationOrder == inputRotationOrder)
        {
          bvh->jointHierarchy[jID].channelRotationOrder = swappedRotationOrder;
          //fprintf(stderr,"swapping jID %u (%s) from %u to %u\n",jID,bvh->jointHierarchy[jID].jointName,inputRotationOrder,swappedRotationOrder);
        }
    }

    return 1;
  }

 return 0;
}


int bvh_swapJointNameRotationAxis(struct BVH_MotionCapture * bvh,const char * jointName,char inputRotationOrder,char swappedRotationOrder)
{
  if ( (bvh!=0) && (bvh->jointHierarchy!=0) )
  {
    BVHJointID jID=0;

    if ( bvh_getJointIDFromJointName(bvh,jointName,&jID) )
    {
      if (bvh->jointHierarchy[jID].channelRotationOrder == inputRotationOrder)
        {
          bvh->jointHierarchy[jID].channelRotationOrder = swappedRotationOrder;
          //fprintf(stderr,"swapping jID %u (%s) from %u to %u\n",jID,bvh->jointHierarchy[jID].jointName,inputRotationOrder,swappedRotationOrder);
          return 1;
        }
    }
  }

 return 0;
}


