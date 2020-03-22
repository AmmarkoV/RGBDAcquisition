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
              int isSelected = 1;

              if (mc->selectedJoints!=0)
              {
                if (!mc->selectedJoints[jID])
                {
                  isSelected=0;
                }
              }

               if (isSelected)
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
            }

       if (numberOfSamples>0)
       {
         return (float)  sumOf2DDistances/numberOfSamples;
       }
     } //-----------------

 return 0.0;
}




float approximateTargetFromMotionBuffer(
                                         struct BVH_MotionCapture * mc,
                                         struct simpleRenderer *renderer,
                                         struct MotionBuffer * solution,
                                         struct BVH_Transform * bvhTargetTransform
                                        )
{
  struct BVH_Transform bvhSourceTransform={0};

  if (
       bvh_loadTransformForMotionBuffer(
                                        mc,
                                        solution->motion,
                                        &bvhSourceTransform
                                       )
     )
     {
               bvh_removeTranslationFromTransform(
                                            mc,
                                            &bvhSourceTransform
                                          );


       return BVH2DDistace(mc,renderer,&bvhSourceTransform,bvhTargetTransform);
     }

 return 0.0;
}




int BVHTestIK(
              struct BVH_MotionCapture * mc,
              unsigned int fIDSource,
              unsigned int fIDTarget
             )
{
  int result=0;

  struct BVH_Transform bvhTargetTransform={0};

  struct simpleRenderer renderer={0};
  simpleRendererDefaults(
                         &renderer,
                         1920, 1080, 582.18394,   582.52915 // https://gopro.com/help/articles/Question_Answer/HERO4-Field-of-View-FOV-Information
                        );
  simpleRendererInitialize(&renderer);

  fprintf(stderr,"BVH file has motion files with %u elements\n",mc->numberOfValuesPerFrame);
  struct MotionBuffer solution={0};
  solution.bufferSize = mc->numberOfValuesPerFrame;
  solution.motion = (float *) malloc(sizeof(float) * (solution.bufferSize+1));

  if (solution.motion!=0)
  {
    if ( bvh_copyMotionFrameToMotionBuffer(mc,&solution,fIDSource) )
    {
      if ( bvh_loadTransformForFrame(mc,fIDTarget,&bvhTargetTransform) )
      {
        bvh_removeTranslationFromTransform(
                                            mc,
                                            &bvhTargetTransform
                                          );

       float error2D = approximateTargetFromMotionBuffer(
                                                         mc,
                                                         &renderer,
                                                         &solution,
                                                         &bvhTargetTransform
                                                        );

        fprintf(stderr,"2D Distance is %0.2f\n",error2D);
        result=1;
      }
    }
    free(solution.motion);
  }

 return result;
}











//./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK 4 100




float approximateTargetPose(
                            struct BVH_MotionCapture * mc,
                            struct simpleRenderer *renderer,
                            struct BVH_Transform * bvhSourceTransform,
                            struct BVH_Transform * bvhTargetTransform
                           )
{
  return  BVH2DDistace(mc,renderer,bvhSourceTransform,bvhTargetTransform);
}



int BVHTestIKOLD(
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

/*
  float * motionBuffer=0;
  int bvh_loadTransformForMotionBuffer(
                                       mc,
                                       motionBuffer,
                                     struct BVH_Transform * bvhTransform
                                    );*/

  if (
       ( bvh_loadTransformForFrame(mc,fIDSource,&bvhSourceTransform) )
        &&
       ( bvh_loadTransformForFrame(mc,fIDTarget,&bvhTargetTransform) )
     )
     {
        float distance2D = approximateTargetPose(mc,&renderer,&bvhSourceTransform,&bvhTargetTransform);

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
