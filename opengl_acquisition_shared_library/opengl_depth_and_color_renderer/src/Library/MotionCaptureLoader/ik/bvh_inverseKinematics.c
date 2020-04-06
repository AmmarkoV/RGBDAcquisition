#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>

#include <time.h>

#include "bvh_inverseKinematics.h"
#include "levmar.h"

#include "../export/bvh_to_svg.h"
#include "../edit/bvh_cut_paste.h"



#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


#define DISCARD_POSITIONAL_COMPONENT 0
const float distance=-150;

unsigned long tickBaseIK = 0;


void clear_line()
{
  fputs("\033[A\033[2K\033[A\033[2K",stdout);
  rewind(stdout);
  int i=ftruncate(1,0);
  if (i!=0) { /*fprintf(stderr,"Error with ftruncate\n");*/ }
}

unsigned long GetTickCountMicrosecondsIK()
{
    struct timespec ts;
    if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0)
        {
            return 0;
        }

    if (tickBaseIK==0)
        {
            tickBaseIK = ts.tv_sec*1000000 + ts.tv_nsec/1000;
            return 0;
        }

    return ( ts.tv_sec*1000000 + ts.tv_nsec/1000 ) - tickBaseIK;
}


unsigned long GetTickCountMillisecondsIK()
{
    return (unsigned long) GetTickCountMicrosecondsIK()/1000;
}


float getSquared3DPointDistance(float aX,float aY,float aZ,float bX,float bY,float bZ)
{
  float diffX = (float) aX-bX;
  float diffY = (float) aY-bY;
  float diffZ = (float) aZ-bZ;
    //We calculate the distance here..!
  return (diffX*diffX) + (diffY*diffY) + (diffZ*diffZ);
}


float get3DPointDistance(float aX,float aY,float aZ,float bX,float bY,float bZ)
{
  return sqrt(getSquared3DPointDistance(aX,aY,aZ,bX,bY,bZ));
}


float getSquared2DPointDistance(float aX,float aY,float bX,float bY)
{
  float diffX = (float) aX-bX;
  float diffY = (float) aY-bY;
    //We calculate the distance here..!
  return (diffX*diffX) + (diffY*diffY);
}


float get2DPointDistance(float aX,float aY,float bX,float bY)
{
  return sqrt(getSquared2DPointDistance(aX,aY,bX,bY));
}


float meanBVH2DDistace(
                       struct BVH_MotionCapture * mc,
                       struct simpleRenderer *renderer,
                       int useAllJoints,
                       BVHMotionChannelID onlyConsiderChildrenOfThisJoint,
                       struct BVH_Transform * bvhSourceTransform,
                       struct BVH_Transform * bvhTargetTransform
                      )
{
   if (bvh_projectTo2D(mc,bvhSourceTransform,renderer,0,0))
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

               if ( (isSelected) && ( (useAllJoints) || (mc->jointHierarchy[jID].parentJoint == onlyConsiderChildrenOfThisJoint) ) )
               {
                 float tX=bvhTargetTransform->joint[jID].pos2D[0];
                 float tY=bvhTargetTransform->joint[jID].pos2D[1];

                 if ( (tX!=0.0) || (tY!=0.0) )
                 {
                  float this2DDistance=get2DPointDistance(
                                                        (float) bvhSourceTransform->joint[jID].pos2D[0],
                                                        (float) bvhSourceTransform->joint[jID].pos2D[1],
                                                        (float) bvhTargetTransform->joint[jID].pos2D[0],
                                                        (float) bvhTargetTransform->joint[jID].pos2D[1]
                                                       );
                  fprintf(stderr,"src(%0.1f,%0.1f)->tar(%0.1f,%0.1f) : ",bvhSourceTransform->joint[jID].pos2D[0],bvhSourceTransform->joint[jID].pos2D[1],bvhTargetTransform->joint[jID].pos2D[0],bvhTargetTransform->joint[jID].pos2D[1]);
                  fprintf(stderr,"2D %s distance = %0.1f\n",mc->jointHierarchy[jID].jointName,this2DDistance);

                  numberOfSamples+=1;
                  sumOf2DDistances+=this2DDistance;
                 }
               }
            }

       if (numberOfSamples>0)
       {
         return (float)  sumOf2DDistances/numberOfSamples;
       }
     } //-----------------

 return 0.0;
}



float meanBVH3DDistace(
                        struct BVH_MotionCapture * mc,
                        struct simpleRenderer *renderer,
                        int useAllJoints,
                        BVHMotionChannelID onlyConsiderChildrenOfThisJoint,
                        float * sourceMotionBuffer,
                        struct BVH_Transform * bvhSourceTransform,
                        float * targetMotionBuffer,
                        struct BVH_Transform * bvhTargetTransform
                      )
{

  if (targetMotionBuffer==0) { return NAN;}

   if (
        (
          performPointProjectionsForMotionBuffer(
                                                 mc,
                                                 bvhSourceTransform,
                                                 sourceMotionBuffer,
                                                 renderer,
                                                 0,
                                                 0
                                                )
        ) &&
        (
          performPointProjectionsForMotionBuffer(
                                                 mc,
                                                 bvhTargetTransform,
                                                 targetMotionBuffer,
                                                 renderer,
                                                 0,
                                                 0
                                                )
        )
      )
      {
       //-----------------
       float sumOf3DDistances=0.0;
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

               if ( (isSelected) && ( (useAllJoints) || (mc->jointHierarchy[jID].parentJoint == onlyConsiderChildrenOfThisJoint) ) )
               {
                 float tX=bvhTargetTransform->joint[jID].pos3D[0];
                 float tY=bvhTargetTransform->joint[jID].pos3D[1];
                 float tZ=bvhTargetTransform->joint[jID].pos3D[2];

                 if ( (tX!=0.0) || (tY!=0.0) || (tZ!=0.0) )
                 {
                  float this3DDistance=get3DPointDistance(
                                                         (float) bvhSourceTransform->joint[jID].pos3D[0],
                                                         (float) bvhSourceTransform->joint[jID].pos3D[1],
                                                         (float) bvhSourceTransform->joint[jID].pos3D[2],
                                                         (float) tX,
                                                         (float) tY,
                                                         (float) tZ
                                                        );

                 fprintf(stderr,"src(%0.1f,%0.1f,%0.1f)->tar(%0.1f,%0.1f,%0.1f) : ",(float) bvhSourceTransform->joint[jID].pos3D[0],
                                                               (float) bvhSourceTransform->joint[jID].pos3D[1],
                                                               (float) bvhSourceTransform->joint[jID].pos3D[2],
                                                               (float) tX,
                                                               (float) tY,
                                                               (float) tZ);
                 fprintf(stderr," %s distance^2 = %0.1f\n",mc->jointHierarchy[jID].jointName,this3DDistance);

                 numberOfSamples+=1;
                 sumOf3DDistances+=this3DDistance;
                }
              }
            }

       if (numberOfSamples>0)
       {
         return (float)  sumOf3DDistances/numberOfSamples;
       }
     } //-----------------

 return 0.0;
}



int cleanProblem(struct ikProblem * problem)
{
  freeMotionBuffer(problem->currentSolution);

  for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
  {
    freeMotionBuffer(problem->chain[chainID].currentSolution);
  }
 return 1;
}

int prepareProblem(
                   struct ikProblem * problem,
                   struct BVH_MotionCapture * mc,
                   struct simpleRenderer *renderer,
                   struct MotionBuffer * solution,
                   struct BVH_Transform * bvhTargetTransform
                  )
{
  problem->mc = mc;
  problem->renderer = renderer;
  problem->initialSolution = solution ;

  problem->currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);

  //2D Projections Targeted
  //----------------------------------------------------------
  problem->bvhTarget2DProjectionTransform = bvhTargetTransform;

 //Chain of subproblems that need to be solved
  //----------------------------------------------------------
  problem->numberOfChains = 5;
  problem->numberOfGroups = 2;
  problem->numberOfJobsPerGroup = 6;



  //Chain #0 is Joint Hip-> to all its children
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------
  unsigned int groupID=0;
  unsigned int jobID=0;
  unsigned int chainID=0;
  unsigned int partID=0;
  BVHJointID thisJID=0;
  //----------------------------------------------------------

  //Chain 0 is the Hip and all of the rigid torso
  //----------------------------------------------------------
  problem->chain[chainID].groupID=groupID;
  problem->chain[chainID].jobID=jobID;
  problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);


  #if DISCARD_POSITIONAL_COMPONENT
   fprintf(stderr,"Ignoring positional component..\n");
  #else
   if (bvh_getJointIDFromJointName(mc,"hip",&thisJID) )
   {
     problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
     problem->chain[chainID].part[partID].jID=thisJID;
     problem->chain[chainID].part[partID].mIDStart=0; //First Rotation
     problem->chain[chainID].part[partID].mIDEnd=2; //First Rotation
     ++partID;
   }
  #endif // DISCARD_POSITIONAL_COMPONENT


  if (bvh_getJointIDFromJointName(mc,"hip",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=3; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=5; //First Rotation
   ++partID;
  } else
  { fprintf(stderr,"No hip in armature..\n"); return 0; }

  if (bvh_getJointIDFromJointName(mc,"neck",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  } else
  { fprintf(stderr,"No neck in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"rshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rShldr",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  } else
  { fprintf(stderr,"No rshoulder in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"lshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lShldr",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  } else
  { fprintf(stderr,"No lshoulder in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"rhip",&thisJID) )  || (bvh_getJointIDFromJointName(mc,"rThigh",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  } else
  { fprintf(stderr,"No rhip in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"lhip",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lThigh",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;

   ++partID;
  } else
  { fprintf(stderr,"No lhip in armature..\n"); return 0; }

 problem->chain[chainID].numberOfParts=partID;

 ++chainID;
 ++groupID;
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------


  //Chain 1 is the Right Arm
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------
  partID=0;
  problem->chain[chainID].groupID=groupID;
  problem->chain[chainID].jobID=jobID;
  problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);



  if ( (bvh_getJointIDFromJointName(mc,"rshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rShldr",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  } else
  { fprintf(stderr,"No rshoulder in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"relbow",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rForeArm",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  } else
  { fprintf(stderr,"No relbow in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"rhand",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rHand",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  } else
  { fprintf(stderr,"No rhand in armature..\n"); return 0; }

 problem->chain[chainID].numberOfParts=partID;
 ++chainID;
 ++jobID;
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------





  //Chain 2 is the Left Arm
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------
  partID=0;
  problem->chain[chainID].groupID=groupID;
  problem->chain[chainID].jobID=jobID;
  problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);


  if ( (bvh_getJointIDFromJointName(mc,"lshoulder",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lShldr",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  } else
  { fprintf(stderr,"No lshoulder in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"lelbow",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lForeArm",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  } else
  { fprintf(stderr,"No lelbow in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"lhand",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lHand",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  } else
  { fprintf(stderr,"No lhand in armature..\n"); return 0; }

 problem->chain[chainID].numberOfParts=partID;
 ++chainID;
 ++jobID;
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------




  //Chain 3 is the Right Leg
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------
  partID=0;
  problem->chain[chainID].groupID=groupID;
  problem->chain[chainID].jobID=jobID;
  problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);


  if ( (bvh_getJointIDFromJointName(mc,"rhip",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rThigh",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  } else
  { fprintf(stderr,"No rhip in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"rknee",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"rShin",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  } else
  { fprintf(stderr,"No rknee in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"rfoot",&thisJID) )  || (bvh_getJointIDFromJointName(mc,"rFoot",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  } else
  { fprintf(stderr,"No rfoot in armature..\n"); return 0; }

 problem->chain[chainID].numberOfParts=partID;
 ++chainID;
 ++jobID;
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------



  //Chain 4 is the Left Leg
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------
  partID=0;
  problem->chain[chainID].groupID=groupID;
  problem->chain[chainID].jobID=jobID;
  problem->chain[chainID].currentSolution=mallocNewMotionBufferAndCopy(mc,problem->initialSolution);

  if ( (bvh_getJointIDFromJointName(mc,"lhip",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lThigh",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  } else
  { fprintf(stderr,"No hip in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"lknee",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lShin",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  } else
  { fprintf(stderr,"No lknee in armature..\n"); return 0; }

  if ( (bvh_getJointIDFromJointName(mc,"lfoot",&thisJID) ) || (bvh_getJointIDFromJointName(mc,"lFoot",&thisJID)) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  } else
  { fprintf(stderr,"No lfoot in armature..\n"); return 0; }

 problem->chain[chainID].numberOfParts=partID;
 ++chainID;
 ++jobID;
  //----------------------------------------------------------
  //----------------------------------------------------------
  //----------------------------------------------------------
  ++groupID;

  problem->numberOfChains = chainID;
  problem->numberOfGroups = groupID;

 return 1;
}


int viewProblem(
                struct ikProblem * problem
               )
{
 fprintf(stderr,"The IK problem we want to solve has %u groups of subproblems\n",problem->numberOfGroups);
 fprintf(stderr,"It is also ultimately divided into %u kinematic chains\n",problem->numberOfChains);

 for (unsigned int chainID=0; chainID<problem->numberOfChains; chainID++)
 {
   fprintf(stderr,"Chain %u has %u parts : ",chainID,problem->chain[chainID].numberOfParts);
   for (unsigned int partID=0; partID<problem->chain[chainID].numberOfParts; partID++)
   {
     unsigned int jID=problem->chain[chainID].part[partID].jID;

     if (problem->chain[chainID].part[partID].endEffector)
     {
      fprintf(
              stderr,"jID(%s/%u)->EndEffector ",
              problem->mc->jointHierarchy[jID].jointName,
              jID
             );
     } else
     {
     fprintf(
             stderr,"jID(%s/%u)->mID(%u to %u) ",
             problem->mc->jointHierarchy[jID].jointName,
             jID,
             problem->chain[chainID].part[partID].mIDStart,
             problem->chain[chainID].part[partID].mIDEnd
             );
     }
   }
   fprintf(stderr,"\n");
 }

 return 1;
}




float calculateChainLoss(
                         struct ikProblem * problem,
                         unsigned int chainID
                        )
{
  unsigned int numberOfSamples=0;
  float loss=0.0;
  if (chainID<problem->numberOfChains)
  {
   //fprintf(stderr,"Chain %u has %u parts : ",chainID,problem->chain[chainID].numberOfParts);

     if (
         bvh_loadTransformForMotionBuffer(
                                          problem->mc,
                                          problem->chain[chainID].currentSolution->motion,
                                          &problem->chain[chainID].current2DProjectionTransform
                                         )
        )
      {

       #if DISCARD_POSITIONAL_COMPONENT
        bvh_removeTranslationFromTransform(
                                            problem->mc,
                                            &problem->chain[chainID].current2DProjectionTransform
                                          );
       #endif // DISCARD_POSITIONAL_COMPONENT

      if (
          (bvh_projectTo2D(problem->mc,&problem->chain[chainID].current2DProjectionTransform,problem->renderer,0,0))
          // && (bvh_projectTo2D(problem->mc,problem->bvhTarget2DProjectionTransform,problem->renderer,0,0))
         )
      {
       for (unsigned int partID=0; partID<problem->chain[chainID].numberOfParts; partID++)
       {
         unsigned int jID=problem->chain[chainID].part[partID].jID;
         float tX = problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[0];
         float tY = problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[1];

         if ((tX!=0.0) || (tY!=0.0) )
         { //Ignore empty joints ..!
          float thisSquared2DDistance=getSquared2DPointDistance(
                                                                (float) problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[0],
                                                                (float) problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[1],
                                                                tX,
                                                                tY
                                                               );
                                                               /*
          fprintf(stderr,"%0.2f,%0.2f -> %0.2f,%0.2f : ",
          problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[0],
          problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[1],
          problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[0],
          problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[1]
          );
          fprintf(stderr,"Joint squared %s distance is %0.2f\n",problem->mc->jointHierarchy[jID].jointName,thisSquared2DDistance);*/
          loss+=thisSquared2DDistance;
          ++numberOfSamples;
         }
       }
     }

      } //Have a valid 2D transform
 } //Have a valid chain

  //I have left 0/0 on purpose to cause NaNs when projection errors occur
  //---------------------------------------------------------------------
  if (numberOfSamples!=0)
   {
     loss = (float) loss/numberOfSamples;
   } else
   {
     loss = NAN;
   }
  //---------------------------------------------------------------------
  return loss;
}


float iteratePartLoss(
                         struct ikProblem * problem,
                         unsigned int chainID,
                         unsigned int partID,
                         float lr,
                         unsigned int epochs
                        )
{
 unsigned int consecutiveBadSteps=0;
 unsigned int mIDS[3];
 mIDS[0]= problem->chain[chainID].part[partID].mIDStart;
 mIDS[1]= problem->chain[chainID].part[partID].mIDStart+1;
 mIDS[2]= problem->chain[chainID].part[partID].mIDStart+2;


 float originalValues[3];
 originalValues[0] = problem->chain[chainID].currentSolution->motion[mIDS[0]];
 originalValues[1] = problem->chain[chainID].currentSolution->motion[mIDS[1]];
 originalValues[2] = problem->chain[chainID].currentSolution->motion[mIDS[2]];

 float previousValues[3];
 previousValues[0] = originalValues[0];
 previousValues[1] = originalValues[1];
 previousValues[2] = originalValues[2];

 float currentValues[3];
 currentValues[0] = originalValues[0];
 currentValues[1] = originalValues[1];
 currentValues[2] = originalValues[2];

 float bestValues[3];
 bestValues[0] = originalValues[0];
 bestValues[1] = originalValues[1];
 bestValues[2] = originalValues[2];

 float previousLoss[3];
 float currentLoss[3];
 float delta[3]={0};

 float initialLoss = calculateChainLoss(problem,chainID);

 previousLoss[0]=initialLoss;
 previousLoss[1]=initialLoss;
 previousLoss[2]=initialLoss;

 float bestLoss = initialLoss;
 float loss=initialLoss;

 unsigned int maximumConsecutiveBadEpochs=4;
 float e=0.001;
 float d=0.01;
 float gradient;
 float distanceFromInitial;
 float spring = 10.0;

 //Give an initial direction..
 delta[0] = d;
 delta[1] = d;
 delta[2] = d;

 currentValues[0] += d;
 currentValues[1] += d;
 currentValues[2] += d;



 fprintf(stderr,"\nOptimizing %s \n",problem->mc->jointHierarchy[problem->chain[chainID].part[partID].jID].jointName);
 fprintf(stderr,"  State |   loss   | rX  |  rY  |  rZ \n");
 fprintf(stderr,"Initial | %0.1f | %0.2f  |  %0.2f  |  %0.2f \n",initialLoss,originalValues[0],originalValues[1],originalValues[2]);
 unsigned long startTime = GetTickCountMicrosecondsIK();

 for (unsigned int i=0; i<epochs; i++)
 {

 //-------------------
   problem->chain[chainID].currentSolution->motion[mIDS[0]] = currentValues[0];
   distanceFromInitial=fabs(currentValues[0] - originalValues[0]);
   currentLoss[0]=calculateChainLoss(problem,chainID) + spring * distanceFromInitial * distanceFromInitial;
   problem->chain[chainID].currentSolution->motion[mIDS[0]] = previousValues[0];
 //-------------------
   problem->chain[chainID].currentSolution->motion[mIDS[1]] = currentValues[1];
   distanceFromInitial=fabs(currentValues[1] - originalValues[1]);
   currentLoss[1]=calculateChainLoss(problem,chainID) + spring * distanceFromInitial * distanceFromInitial;
   problem->chain[chainID].currentSolution->motion[mIDS[1]] = previousValues[1];
 //-------------------
   problem->chain[chainID].currentSolution->motion[mIDS[2]] = currentValues[2];
   distanceFromInitial=fabs(currentValues[2] - originalValues[2]);
   currentLoss[2]=calculateChainLoss(problem,chainID) + spring * distanceFromInitial * distanceFromInitial;
   problem->chain[chainID].currentSolution->motion[mIDS[2]] = previousValues[2];
 //-------------------


   //We multiply by 0.5 to do a "One Half Mean Squared Error"
   gradient =  (float) 0.5 * (previousLoss[0] - currentLoss[0]) / (delta[0]+e);
   delta[0] += (float) lr *  gradient;

   gradient =  (float) 0.5 * (previousLoss[1] - currentLoss[1]) / (delta[1]+e);
   delta[1] += (float) lr * gradient;

   gradient =  (float) 0.5 * (previousLoss[2] - currentLoss[2]) / (delta[2]+e);
   delta[2] += (float) lr * gradient;

   previousLoss[0]=currentLoss[0];
   previousLoss[1]=currentLoss[1];
   previousLoss[2]=currentLoss[2];

   previousValues[0]=currentValues[0];
   previousValues[1]=currentValues[1];
   previousValues[2]=currentValues[2];

   currentValues[0]+=delta[0];
   currentValues[1]+=delta[1];
   currentValues[2]+=delta[2];

   problem->chain[chainID].currentSolution->motion[mIDS[0]] = currentValues[0];
   problem->chain[chainID].currentSolution->motion[mIDS[1]] = currentValues[1];
   problem->chain[chainID].currentSolution->motion[mIDS[2]] = currentValues[2];
   loss=calculateChainLoss(problem,chainID);


   if ( (loss<bestLoss) && ( fabs(currentValues[0])<360 ) && ( fabs(currentValues[1])<360 ) && ( fabs(currentValues[2])<360 ) )
   {
     bestLoss=loss;
     bestValues[0]=currentValues[0];
     bestValues[1]=currentValues[1];
     bestValues[2]=currentValues[2];
     consecutiveBadSteps=0;
     fprintf(stderr,"%07u | %0.1f | %0.2f  |  %0.2f  |  %0.2f \n",i,loss,currentValues[0],currentValues[1],currentValues[2]);
   } else
   {
     ++consecutiveBadSteps;
     fprintf(stderr,YELLOW "%07u | %0.1f | %0.2f  |  %0.2f  |  %0.2f \n" NORMAL,i,loss,currentValues[0],currentValues[1],currentValues[2]);
   }



   if (consecutiveBadSteps>=maximumConsecutiveBadEpochs) { fprintf(stderr,YELLOW "Early Stopping\n" NORMAL); break; }
 }
 unsigned long endTime = GetTickCountMicrosecondsIK();


  fprintf(stderr,"Improved loss from %0.2f to %0.2f ( %0.2f%% ) in %lu microseconds \n",initialLoss,bestLoss, 100 - ( (float) 100* bestLoss/initialLoss ),endTime-startTime);
  fprintf(stderr,"Optimized values changed from %0.2f,%0.2f,%0.2f to %0.2f,%0.2f,%0.2f\n",originalValues[0],originalValues[1],originalValues[2],bestValues[0],bestValues[1],bestValues[2]);
  fprintf(stderr,"correction of %0.2f,%0.2f,%0.2f deg\n",bestValues[0]-originalValues[0],bestValues[1]-originalValues[1],bestValues[2]-originalValues[2]);

   problem->chain[chainID].currentSolution->motion[mIDS[0]] = bestValues[0];
   problem->chain[chainID].currentSolution->motion[mIDS[1]] = bestValues[1];
   problem->chain[chainID].currentSolution->motion[mIDS[2]] = bestValues[2];

  return bestLoss;

}








float iterateChainLoss(
                         struct ikProblem * problem,
                         unsigned int chainID,
                         float lr,
                         unsigned int epochs
                        )
{

 copyMotionBuffer(problem->chain[chainID].currentSolution,problem->currentSolution);

 for (unsigned int partID=0; partID<problem->chain[chainID].numberOfParts; partID++)
 {
   if (!problem->chain[chainID].part[partID].endEffector)
   {
    iteratePartLoss(
                    problem,
                    chainID,
                    partID,
                    lr,
                    epochs
                   );
   }
 }

 copyMotionBuffer(problem->currentSolution,problem->chain[chainID].currentSolution);

 return calculateChainLoss(problem,chainID);
}







int approximateBodyFromMotionBufferUsingInverseKinematics(
                                         struct BVH_MotionCapture * mc,
                                         struct simpleRenderer *renderer,
                                         struct MotionBuffer * solution,
                                         float learningRate,
                                         unsigned int iterations,
                                         unsigned int epochs,
                                         struct MotionBuffer * groundTruth,
                                         struct BVH_Transform * bvhTargetTransform,
                                         float * initialMAEInPixels,
                                         float * finalMAEInPixels,
                                         float * initialMAEInMM,
                                         float * finalMAEInMM,
                                         int dumpScreenshots
                                        )
{

 struct ikProblem problem={0};

  if (!prepareProblem(
                      &problem,
                      mc,
                      renderer,
                      solution,
                      bvhTargetTransform
                     )
     )
     {
        fprintf(stderr,"Could not prepare the problem for IK solution\n");
        return 0;
     }

  viewProblem(&problem);



  //---------------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------------
  struct BVH_Transform bvhCurrentTransform={0};


  if (
         bvh_loadTransformForMotionBuffer(
                                          mc,
                                          solution->motion,
                                          &bvhCurrentTransform
                                         )
        )
      {
       #if DISCARD_POSITIONAL_COMPONENT
        bvh_removeTranslationFromTransform(
                                            mc,
                                            &bvhCurrentTransform
                                          );
       #endif // DISCARD_POSITIONAL_COMPONENT


      //----------------------------------------------------
       if (initialMAEInPixels!=0)
       {
        *initialMAEInPixels = meanBVH2DDistace(
                                       mc,
                                       renderer,
                                       1,
                                       0,
                                       &bvhCurrentTransform,
                                       bvhTargetTransform
                                      );
       }
      //----------------------------------------------------

      if ( (initialMAEInMM!=0) && (groundTruth!=0) )
       {
        *initialMAEInMM = meanBVH3DDistace(
                                       mc,
                                       renderer,
                                       1,
                                       0,
                                       solution->motion,
                                       &bvhCurrentTransform,
                                       groundTruth->motion,
                                       bvhTargetTransform
                                      );
       }
      //----------------------------------------------------

      }

  if (dumpScreenshots)
  {
   dumpBVHToSVGFrame(
                     "initial.svg",
                      mc,
                      &bvhCurrentTransform,
                      0,
                      renderer
                     );
  }
  //---------------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------------




  float loss;

  for (int t=0; t<iterations; t++)
  {
   loss = iterateChainLoss(&problem,0,learningRate,epochs);
   loss = iterateChainLoss(&problem,1,learningRate,epochs);
   loss = iterateChainLoss(&problem,2,learningRate,epochs);
   loss = iterateChainLoss(&problem,3,learningRate,epochs);
   loss = iterateChainLoss(&problem,4,learningRate,epochs);
  }

   copyMotionBuffer(solution,problem.currentSolution);






  //---------------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------------

  if (
         bvh_loadTransformForMotionBuffer(
                                          mc,
                                          solution->motion,
                                          &bvhCurrentTransform
                                         )
     )
      {
       #if DISCARD_POSITIONAL_COMPONENT
        bvh_removeTranslationFromTransform(
                                            mc,
                                            &bvhCurrentTransform
                                          );
       #endif // DISCARD_POSITIONAL_COMPONENT


      //----------------------------------------------------
      if (finalMAEInPixels!=0)
       {
        *finalMAEInPixels  = meanBVH2DDistace(
                                       mc,
                                       renderer,
                                       1,
                                       0,
                                       &bvhCurrentTransform,
                                       bvhTargetTransform
                                     );
      }
      //----------------------------------------------------
      if ( (finalMAEInMM!=0) && (groundTruth!=0) )
       {
        *finalMAEInMM = meanBVH3DDistace(
                                       mc,
                                       renderer,
                                       1,
                                       0,
                                       solution->motion,
                                       &bvhCurrentTransform,
                                       groundTruth->motion,
                                       bvhTargetTransform
                                      );
      }
      //----------------------------------------------------


      }
  //---------------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------------
  //---------------------------------------------------------------------------------------

  if (dumpScreenshots)
  {
  dumpBVHToSVGFrame(
                     "target.svg",
                      mc,
                      bvhTargetTransform,
                      1,
                      renderer
                     );

  dumpBVHToSVGFrame(
                     "solution.svg",
                      mc,
                      &bvhCurrentTransform,
                      0,
                      renderer
                     );

 }


 //Cleanup allocations needed for the problem..
 cleanProblem(&problem);

 return 1;
}


int writeHTML(
               unsigned int fIDSource,
               unsigned int fIDTarget,
               float initialMAEInPixels,
               float finalMAEInPixels,
               float initialMAEInMM,
               float finalMAEInMM,
               int dumpScreenshots
             )
{
  if (dumpScreenshots)
  {
  int i=system("convert initial.svg initial.png&");
  if (i!=0) { fprintf(stderr,"Error converting image..\n"); }
  i=system("convert target.svg target.png&");
  if (i!=0) { fprintf(stderr,"Error converting image..\n"); }
  i=system("convert solution.svg solution.png&");
  if (i!=0) { fprintf(stderr,"Error converting image..\n"); }

  FILE * html=fopen("report.html","w");
  if (html!=0)
  {
    fprintf(html,"<html><body><br>\n");
    //------------------------------------------------------------
    fprintf(html,"<table><tr>\n");
    fprintf(html,"<td><img src=\"initial.png\" width=400></td>\n");
    fprintf(html,"<td><img src=\"target.png\" width=400></td>\n");
    fprintf(html,"<td><img src=\"solution.png\" width=400></td>\n");
    fprintf(html,"</tr>\n");
    //------------------------------------------------------------
    fprintf(html,"<tr>\n");
    fprintf(html,"<td align=\"center\">Initial Pose (frame %u)</td>\n",fIDSource);
    fprintf(html,"<td align=\"center\">Target Pose (frame %u)</td>\n",fIDTarget);
    fprintf(html,"<td align=\"center\">Solution</td>\n");
    fprintf(html,"</tr>\n");
    fprintf(html,"</table>\n");
    //------------------------------------------------------------

    fprintf(html,"<br><br><br><br>");
    fprintf(html,"MAE in 2D Pixels went from %0.2f to %0.2f <br>",initialMAEInPixels,finalMAEInPixels);
    fprintf(html,"MAE in 3D mm went from %0.2f to %0.2f <br>",initialMAEInMM*10,finalMAEInMM*10);

    fprintf(html,"</body></html>");
    fclose(html);
    return 1;
  }

 }
  return 0;
}


//./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK 4 100

int bvhTestIK(
              struct BVH_MotionCapture * mc,
              float lr,
              unsigned int iterations,
              unsigned int epochs,
              unsigned int fIDSource,
              unsigned int fIDTarget
             )
{
  int result=0;

  int dumpScreenshots = 1;

  struct BVH_Transform bvhTargetTransform={0};

  struct simpleRenderer renderer={0};
  simpleRendererDefaults( &renderer, 1920, 1080, 582.18394,   582.52915 );// https://gopro.com/help/articles/Question_Answer/HERO4-Field-of-View-FOV-Information
  simpleRendererInitialize(&renderer);

  fprintf(stderr,"BVH file has motion files with %u elements\n",mc->numberOfValuesPerFrame);

  float initialMAEInPixels=0.0,finalMAEInPixels=0.0;
  float initialMAEInMM=0.0,finalMAEInMM=0.0;

  //Load all motion buffers
  struct MotionBuffer * groundTruth     = mallocNewMotionBuffer(mc);
  struct MotionBuffer * initialSolution = mallocNewMotionBuffer(mc);
  struct MotionBuffer * solution        = mallocNewMotionBuffer(mc);


  if ( (solution!=0) && (groundTruth!=0) && (initialSolution!=0) )
  {
    if (
          ( bvh_copyMotionFrameToMotionBuffer(mc,initialSolution,fIDSource) ) &&
          ( bvh_copyMotionFrameToMotionBuffer(mc,solution,fIDSource) ) &&
          ( bvh_copyMotionFrameToMotionBuffer(mc,groundTruth,fIDTarget) )
       )
      {
       initialSolution->motion[0]=0;
       initialSolution->motion[1]=0;
       initialSolution->motion[2]=distance;


       solution->motion[0]=0;
       solution->motion[1]=0;
       solution->motion[2]=distance;


       groundTruth->motion[0]=0;
       groundTruth->motion[1]=0;
       groundTruth->motion[2]=distance;


        if ( bvh_loadTransformForMotionBuffer(mc,groundTruth->motion,&bvhTargetTransform) )
         {
           if  (bvh_projectTo2D(mc,&bvhTargetTransform,&renderer,0,0))
           {
            #if DISCARD_POSITIONAL_COMPONENT
            bvh_removeTranslationFromTransform(
                                                mc,
                                                &bvhTargetTransform
                                              );
            #endif // DISCARD_POSITIONAL_COMPONENT



            if (
                approximateBodyFromMotionBufferUsingInverseKinematics(
                                                                       mc,
                                                                       &renderer,
                                                                       solution,
                                                                       lr,
                                                                       iterations,
                                                                       epochs,
                                                                       groundTruth,
                                                                       &bvhTargetTransform,
                                                                       &initialMAEInPixels,
                                                                       &finalMAEInPixels,
                                                                       &initialMAEInMM,
                                                                       &finalMAEInMM,
                                                                       dumpScreenshots
                                                                      )
                )
            {
             //=======
             result=1;
             //=======


            //-------------------------------------------------------------------------------------------------------------
            //compareMotionBuffers("The problem we want to solve compared to the initial state",initialSolution,groundTruth);
            //compareMotionBuffers("The solution we proposed compared to ground truth",solution,groundTruth);
            //-------------------------------------------------------------------------------------------------------------
            compareTwoMotionBuffers(mc,"Improvement",initialSolution,solution,groundTruth);

            fprintf(stderr,"MAE in 2D Pixels went from %0.2f to %0.2f \n",initialMAEInPixels,finalMAEInPixels);
            fprintf(stderr,"MAE in 3D mm went from %0.2f to %0.2f \n",initialMAEInMM*10,finalMAEInMM*10);


             //-------------------------------------------------------------------------------------------------
             writeHTML(
                       fIDSource,
                       fIDTarget,
                       initialMAEInPixels,
                       finalMAEInPixels,
                       initialMAEInMM,
                       finalMAEInMM,
                       dumpScreenshots
                     );
             //-------------------------------------------------------------------------------------------------
            } else
            {
              fprintf(stderr,"Failed to run IK code..\n");
            }


           } else
            {
              fprintf(stderr,"Could not project 2D points of target..\n");
            }
         }
      }
    freeMotionBuffer(solution);
    freeMotionBuffer(initialSolution);
    freeMotionBuffer(groundTruth);
  }

 return result;
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
  fprintf(stderr,"NOT IMPLEMENTED YET..");
  //Todo mirror 2D points in 2D and then perform IK..
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
