#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>


#include "bvh_cut_paste.h"
#include "bvh_inverseKinematics.h"
#include "bvh_cut_paste.h"

float getSquared2DPointDistance(float aX,float aY,float bX,float bY)
{
  float diffX = (float) aX-bX;
  float diffY = (float) aY-bY;
    //We calculate the distance here..!
  return (diffX*diffX)+(diffY*diffY);
}

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
                float thisSquared2DDistance=getSquared2DPointDistance(
                                                                      (float) bvhSourceTransform->joint[jID].pos2D[0],
                                                                      (float) bvhSourceTransform->joint[jID].pos2D[1],
                                                                      (float) bvhTargetTransform->joint[jID].pos2D[0],
                                                                      (float) bvhTargetTransform->joint[jID].pos2D[1]
                                                                     );
               fprintf(stderr,"%0.2f,%0.2f -> %0.2f,%0.2f : ",bvhSourceTransform->joint[jID].pos2D[0],bvhSourceTransform->joint[jID].pos2D[1],bvhTargetTransform->joint[jID].pos2D[0],bvhTargetTransform->joint[jID].pos2D[1]);
               fprintf(stderr,"Joint squared %s distance is %0.2f\n",mc->jointHierarchy[jID].jointName,thisSquared2DDistance);

               numberOfSamples+=1;
               sumOf2DDistances+=thisSquared2DDistance;
              }
            }

       if (numberOfSamples>0)
       {
         return (float)  sumOf2DDistances/numberOfSamples;
       }
     } //-----------------

 return 0.0;
}



void clear_line()
{
  fputs("\033[A\033[2K\033[A\033[2K",stdout);
  rewind(stdout);
  int i=ftruncate(1,0);
  if (i!=0) { /*fprintf(stderr,"Error with ftruncate\n");*/ }
}


int bruteForceChange(
                     struct BVH_MotionCapture * mc,
                     struct simpleRenderer *renderer,
                     struct MotionBuffer * solution,
                     unsigned int fromElement,
                     unsigned int toElement,
                     unsigned int budget,
                     struct BVH_Transform * bvhTargetTransform
                    )
{
  unsigned int degreesOfFreedomForTheProblem = toElement - fromElement + 1;
  unsigned int budgetPerDoF=(unsigned int) budget/degreesOfFreedomForTheProblem;
  fprintf(stdout,"Trying to solve a %u D.o.F. problem with a budget of %u tries..\n",degreesOfFreedomForTheProblem,budget);


  char jointName[256]={0};

  for (BVHMotionChannelID mID=fromElement; mID<toElement+1; mID++)
  {
    if (bvh_getMotionChannelName(mc,mID,jointName,256))
    {
     fprintf(stdout,"%s ",jointName);
    } else
    {
     fprintf(stdout,"mID=%u ",mID);
    }
  }
  fprintf(stdout,"\n______________________\n");


  for (BVHMotionChannelID mID=fromElement; mID<toElement+1; mID++)
  {
    for (int i=0; i<budgetPerDoF; i++)
    {

    }
  }


 return 1;
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

        bruteForceChange(
                          mc,
                          renderer,
                          solution,
                          3,
                          5,
                          100,
                          bvhTargetTransform
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
