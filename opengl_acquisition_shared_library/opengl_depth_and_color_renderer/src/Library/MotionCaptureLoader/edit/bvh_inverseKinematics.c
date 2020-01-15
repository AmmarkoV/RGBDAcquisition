#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bvh_inverseKinematics.h"
#include "bvh_cut_paste.h"



float get2DPointDistance(float aX,float aY,float bX,float bY)
{
  float diffX = (float) aX-bX;
  float diffY = (float) aY-bY;
    //We calculate the distance here..!
  return sqrt((diffX*diffX)+(diffY*diffY));
}

float BVH2DDistace(
                   struct BVH_MotionCapture * mc,
                   struct simpleRenderer *renderer,
                   struct BVH_Transform * bvhSourceTransform,
                   struct BVH_Transform * bvhTargetTransform
                  )
{

   if ( 
        (bvh_projectTo2D(mc,bvhSourceTransform,renderer,0,0)) &&
        (bvh_projectTo2D(mc,bvhTargetTransform,renderer,0,0)) 
      )
      {
       //-----------------
       float sumOf2DDistances=0.0;
       unsigned int numberOfSamples=0;
       for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
            {
              float this2DDistance=get2DPointDistance(
                                                      (float) bvhSourceTransform->joint[jID].pos2D[0],
                                                      (float) bvhSourceTransform->joint[jID].pos2D[1],
                                                      (float) bvhTargetTransform->joint[jID].pos2D[0],
                                                      (float) bvhTargetTransform->joint[jID].pos2D[1]
                                                     ); 
              fprintf(stderr,"Joint %s distance is %0.2f\n",mc->jointHierarchy[jID].jointName,this2DDistance);

              numberOfSamples+=1;
              sumOf2DDistances+=this2DDistance; 
            }
            
       if (numberOfSamples>0)
       {
         return (float)  sumOf2DDistances/numberOfSamples;
       }     
     } //-----------------
    
 return 0.0;
}




int BVHTestIK(
              struct BVH_MotionCapture * mc,
              unsigned int fIDSource,
              unsigned int fIDTarget
             )
{
  struct BVH_Transform bvhSourceTransform={0};
  struct BVH_Transform bvhTargetTransform={0};

  struct simpleRenderer renderer={0};
  simpleRendererDefaults(
                         &renderer,
                         1920, 1080, 582.18394,   582.52915 // https://gopro.com/help/articles/Question_Answer/HERO4-Field-of-View-FOV-Information
                        );
  simpleRendererInitialize(&renderer);

  if (
       ( bvh_loadTransformForFrame(mc,fIDSource,&bvhSourceTransform) )
        &&
       ( bvh_loadTransformForFrame(mc,fIDTarget,&bvhTargetTransform) )
     )
     {
        float distance2D = BVH2DDistace(mc,&renderer,&bvhSourceTransform,&bvhTargetTransform);

        fprintf(stderr,"2D Distance is %0.2f\n",distance2D);
        return 1;
     }

   return 0;
}


































//https://www.gamasutra.com/blogs/LuisBermudez/20170804/303066/3_Simple_Steps_to_Implement_Inverse_Kinematics.php
//https://groups.csail.mit.edu/drl/journal_club/papers/033005/buss-2004.pdf
//https://simtk-confluence.stanford.edu/display/OpenSim/How+Inverse+Kinematics+Works
int mirrorBVHThroughIK(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform * bvhTransform,
                       unsigned int fID,
                       struct simpleRenderer * renderer,
                       BVHJointID jIDA,
                       BVHJointID jIDB
                      )
{
   float * motionBuffer=0;

   bvh_loadTransformForMotionBuffer(
                                     mc,
                                     motionBuffer,
                                     bvhTransform
                                   );


  //TODO : TODO: TODO:
  //TODO: add here..

  if ( performPointProjectionsForFrame(mc,bvhTransform,fID,renderer,0,0) )
     {
        fprintf(stderr,"Not Implemented, Todo: mirrorBVHThroughIK %u \n",fID);

        fprintf(stderr,"%u=>%0.2f,%0.2f,%0.2f,%0.2f",
                jIDA,
                bvhTransform->joint[jIDA].pos3D[0],
                bvhTransform->joint[jIDA].pos3D[1],
                bvhTransform->joint[jIDA].pos3D[2],
                bvhTransform->joint[jIDA].pos3D[3]
                );

        fprintf(stderr,"  %u=>%0.2f,%0.2f,%0.2f,%0.2f\n",
                jIDB,
                bvhTransform->joint[jIDB].pos3D[0],
                bvhTransform->joint[jIDB].pos3D[1],
                bvhTransform->joint[jIDB].pos3D[2],
                bvhTransform->joint[jIDB].pos3D[3]
                );


     }
  return 0;
}




int bvh_MirrorJointsThroughIK(
                               struct BVH_MotionCapture * mc,
                               const char * jointNameA,
                               const char * jointNameB
                             )
{
  BVHJointID jIDA,jIDB;

  if (
       (!bvh_getJointIDFromJointNameNocase(mc,jointNameA,&jIDA)) ||
       (!bvh_getJointIDFromJointNameNocase(mc,jointNameB,&jIDB))
     )
  {
    fprintf(stderr,"bvh_MirrorJointsThroughIK error resolving joints (%s,%s) \n",jointNameA,jointNameB);fprintf(stderr,"Full list of joints is : \n");
    unsigned int jID=0;
     for (jID=0; jID<mc->jointHierarchySize; jID++)
      {
        fprintf(stderr,"   joint %u = %s\n",jID,mc->jointHierarchy[jID].jointName);
      }
    return 0;
  }


 struct BVH_Transform bvhTransform={0};
 struct simpleRenderer renderer={0};
 simpleRendererDefaults(
                        &renderer,
                        1920, 1080, 582.18394,   582.52915 // https://gopro.com/help/articles/Question_Answer/HERO4-Field-of-View-FOV-Information
                       );
 simpleRendererInitialize(&renderer);

 BVHFrameID fID=0;
 for (fID=0; fID<mc->numberOfFrames; fID++)
         {
            mirrorBVHThroughIK(
                                mc,
                                &bvhTransform,
                                fID,
                                &renderer,
                                jIDA,
                                jIDB
                               );
         }


 return 1;
}
