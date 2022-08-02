#include "bvh_remapangles.h"
#include "../calculate/bvh_transform.h"
#include "../ik/bvh_inverseKinematics.h"

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




int bvh_studyMID2DImpact(
                           struct BVH_MotionCapture * bvh,
                           struct BVH_RendererConfiguration* renderingConfiguration,
                           BVHFrameID fID,
                           BVHMotionChannelID mIDRelativeToOneFrame,
                           float *rangeMinimum,
                           float *rangeMaximum
                          )
{
  struct simpleRenderer renderer={0};
  //Declare and populate the simpleRenderer that will project our 3D points

  if (renderingConfiguration->isDefined)
  {
    renderer.fx     = renderingConfiguration->fX;
    renderer.fy     = renderingConfiguration->fY;
    renderer.skew   = 1.0;
    renderer.cx     = renderingConfiguration->cX;
    renderer.cy     = renderingConfiguration->cY;
    renderer.near   = renderingConfiguration->near;
    renderer.far    = renderingConfiguration->far;
    renderer.width  = renderingConfiguration->width;
    renderer.height = renderingConfiguration->height;

    simpleRendererInitializeFromExplicitConfiguration(&renderer);
    //bvh_freeTransform(&bvhTransform);

    fprintf(stderr,"Direct Rendering is not implemented yet, please don't use it..\n");
    return 0;
  } else
  {
   //This is the normal rendering where we just simulate our camera center
   simpleRendererDefaults(
                          &renderer,
                          renderingConfiguration->width,
                          renderingConfiguration->height,
                          renderingConfiguration->fX,
                          renderingConfiguration->fY
                         );
    simpleRendererInitialize(&renderer);
  }

  FILE * fp = fopen("study.dat","w");

  if (fp!=0)
  {
   struct BVH_Transform bvhTransformOriginal = {0};
   struct BVH_Transform bvhTransformChanged  = {0};


   BVHMotionChannelID mID = fID * bvh->numberOfValuesPerFrame + mIDRelativeToOneFrame;

   fprintf(stderr,"bvh_studyMID2DImpact(%u,%u,%0.2f,%0.2f)\n",fID,mIDRelativeToOneFrame,*rangeMinimum,*rangeMaximum);

   fprintf(stderr,"MID %u / Joint %u/ Channel %u / %s\n",
           mID,
           bvh->motionToJointLookup[mIDRelativeToOneFrame].jointID,
           bvh->motionToJointLookup[mIDRelativeToOneFrame].channelID
           ,bvh->jointHierarchy[bvh->motionToJointLookup[mIDRelativeToOneFrame].jointID].jointName
          );

   bvh_loadTransformForFrame(
                             bvh,
                             fID ,
                             &bvhTransformOriginal,
                             0
                            );

  if (
      bvh_projectTo2D(
                      bvh,
                      &bvhTransformOriginal,
                      &renderer,
                      0, //occlusions,
                      0//directRendering
                    )
      )
      {
        BVHMotionChannelID originalValue = bvh_getMotionValue(bvh,mID);

        float v = *rangeMinimum;
        while (v<*rangeMaximum)
        {
          //fprintf(stderr,"Studying MID => %u / Value %f\n",mID,v);
          bvh_setMotionValue(bvh,mID,v);

          bvh_loadTransformForFrame(
                                    bvh,
                                    fID ,
                                    &bvhTransformChanged,
                                    0
                                   );

          if (
                bvh_projectTo2D(
                                bvh,
                                &bvhTransformChanged,
                                &renderer,
                                0, //occlusions,
                                0//directRendering
                               )
              )
              {
                  float mae =  meanBVH2DDistance(
                                                  bvh,
                                                  &renderer,
                                                  1,
                                                  0,
                                                  &bvhTransformChanged,
                                                  &bvhTransformOriginal,
                                                  0
                                                );
                  //fprintf(stderr," %f\n",mae);
                  fprintf(fp,"%f %f\n",v,mae);
              } else
              {
                fprintf(stderr,"Failed projecting v=%f..\n",v);
              }

          v+=1.0;
        }


          bvh_setMotionValue(bvh,mID,originalValue);
      } else
      {
          fprintf(stderr,"Failed projecting original..\n");
      }
      //using ls 1 t 'TTT'
      system("gnuplot -e \"set terminal png size 1000,800 font 'Helvetica,24'; set output 'out.png'; set style line 1 lt 1 lc rgb 'green' lw 3; plot 'study.dat'  with lines \"");
      fclose(fp);
  }
return 1;
}


