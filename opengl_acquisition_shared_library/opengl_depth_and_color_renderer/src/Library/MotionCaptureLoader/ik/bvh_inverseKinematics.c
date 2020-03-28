#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#include <time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <unistd.h>


#include "bvh_inverseKinematics.h"
#include "levmar.h"

#include "../edit/bvh_cut_paste.h"

#define MAXIMUM_CHAINS 10
#define MAXIMUM_PARTS_OF_CHAIN 10

struct ikChainParts
{
 BVHJointID jID;
 BVHMotionChannelID mIDStart;
 BVHMotionChannelID mIDEnd;
 char evaluated;
 char endEffector;
};
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------

struct ikChain
{
  unsigned int jobID;
  unsigned int groupID;

  unsigned int numberOfParts;
  struct ikChainParts part[MAXIMUM_PARTS_OF_CHAIN];

  struct MotionBuffer * currentSolution;
  struct BVH_Transform current2DProjectionTransform;

  float initialError;
  float previousError;
  float currentError;
};
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------

struct ikProblem
{
 //BVH file that reflects our problem
 struct BVH_MotionCapture * mc;
 //Renderer that handles 3D projections
 struct simpleRenderer *renderer;


 //Initial solution
 struct MotionBuffer * initialSolution;
 //Current solution
 struct MotionBuffer * currentSolution;

 //2D Projections Targeted
 struct BVH_Transform * bvhTarget2DProjectionTransform;

 //Chain of subproblems that need to be solved
 unsigned int numberOfChains;
 unsigned int numberOfGroups;
 unsigned int numberOfJobsPerGroup;

 struct ikChain chain[MAXIMUM_CHAINS];
};
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------

void freeSolutionBuffer(struct MotionBuffer * mb)
{
 if (mb!=0)
 {
  if(mb->motion!=0)
   {
     free(mb->motion);
     mb->motion=0;
   }
  free(mb);
 }
}
//---------------------------------------------------------


struct MotionBuffer * mallocNewSolutionBuffer(struct BVH_MotionCapture * mc)
{
  struct MotionBuffer * newBuffer = (struct MotionBuffer *)  malloc(sizeof(struct MotionBuffer));
  if (newBuffer!=0)
  {
    newBuffer->bufferSize = mc->numberOfValuesPerFrame;
    newBuffer->motion = (float *) malloc(sizeof(float) * newBuffer->bufferSize);
    if (newBuffer->motion!=0)
    {
      memset(newBuffer->motion,0,sizeof(float) * newBuffer->bufferSize);
    }
  }

  return newBuffer;
}
//---------------------------------------------------------


struct MotionBuffer * mallocNewSolutionBufferAndCopy(struct BVH_MotionCapture * mc,struct MotionBuffer * whatToCopy)
{
  struct MotionBuffer * newBuffer = (struct MotionBuffer *)  malloc(sizeof(struct MotionBuffer));
  if (newBuffer!=0)
  {
    newBuffer->bufferSize = mc->numberOfValuesPerFrame;
    newBuffer->motion = (float *) malloc(sizeof(float) * newBuffer->bufferSize);
    if (newBuffer->motion!=0)
    {
      for (unsigned int i=0; i<newBuffer->bufferSize; i++)
      {
        newBuffer->motion[i]=whatToCopy->motion[i];
      }
    }
  }

  return newBuffer;
}
//---------------------------------------------------------



void clear_line()
{
  fputs("\033[A\033[2K\033[A\033[2K",stdout);
  rewind(stdout);
  int i=ftruncate(1,0);
  if (i!=0) { /*fprintf(stderr,"Error with ftruncate\n");*/ }
}

float getSquared2DPointDistance(float aX,float aY,float bX,float bY)
{
  float diffX = (float) aX-bX;
  float diffY = (float) aY-bY;
    //We calculate the distance here..!
  return (diffX*diffX)+(diffY*diffY);
}


float get2DPointDistance(float aX,float aY,float bX,float bY)
{
  return sqrt(getSquared2DPointDistance(aX,aY,bX,bY));
}



int prepareProblem(
                   struct ikProblem * problem,
                   struct BVH_MotionCapture * mc,
                   struct simpleRenderer *renderer,
                   struct MotionBuffer * solution,
                   float * averageError,
                   struct BVH_Transform * bvhTargetTransform
                  )
{
  problem->mc = mc;
  problem->renderer = renderer;
  problem->initialSolution = solution ;

  problem->currentSolution = mallocNewSolutionBuffer(mc);

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
  problem->chain[chainID].currentSolution=mallocNewSolutionBufferAndCopy(mc,problem->initialSolution);

  if (bvh_getJointIDFromJointName(mc,"hip",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=3; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=5; //First Rotation
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"neck",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"rshoulder",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"lshoulder",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"rhip",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"lhip",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;

   ++partID;
  }

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
  problem->chain[chainID].currentSolution=mallocNewSolutionBufferAndCopy(mc,problem->initialSolution);



  if (bvh_getJointIDFromJointName(mc,"rshoulder",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"relbow",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"rhand",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  }

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
  problem->chain[chainID].currentSolution=mallocNewSolutionBufferAndCopy(mc,problem->initialSolution);



  if (bvh_getJointIDFromJointName(mc,"lshoulder",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"lelbow",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"lhand",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  }

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
  problem->chain[chainID].currentSolution=mallocNewSolutionBufferAndCopy(mc,problem->initialSolution);


  if (bvh_getJointIDFromJointName(mc,"rhip",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"rknee",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"rfoot",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  }

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
  problem->chain[chainID].currentSolution=mallocNewSolutionBufferAndCopy(mc,problem->initialSolution);

  if (bvh_getJointIDFromJointName(mc,"lhip",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"lknee",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].endEffector=0;
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].mIDStart=mc->jointToMotionLookup[thisJID].jointMotionOffset; //First Rotation
   problem->chain[chainID].part[partID].mIDEnd=problem->chain[chainID].part[partID].mIDStart + mc->jointHierarchy[thisJID].loadedChannels-1;
   ++partID;
  }

  if (bvh_getJointIDFromJointName(mc,"lfoot",&thisJID) )
  {
   problem->chain[chainID].part[partID].evaluated=0; //Not evaluated yet
   problem->chain[chainID].part[partID].jID=thisJID;
   problem->chain[chainID].part[partID].endEffector=1;
   ++partID;
  }

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


float meanSquaredBVH2DDistace(
                              struct BVH_MotionCapture * mc,
                              struct simpleRenderer *renderer,
                              int useAllJoints,
                              BVHMotionChannelID onlyConsiderChildrenOfThisJoint,
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

               if ( (isSelected) && ( (useAllJoints) || (mc->jointHierarchy[jID].parentJoint == onlyConsiderChildrenOfThisJoint) ) )
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




float calculateChainLoss(
                         struct ikProblem * problem,
                         unsigned int chainID
                        )
{
  unsigned int numberOfSamples=0;
  float loss=0;
  if (chainID<problem->numberOfChains)
  {
   fprintf(stderr,"Chain %u has %u parts : ",chainID,problem->chain[chainID].numberOfParts);

     if (
         bvh_loadTransformForMotionBuffer(
                                          problem->mc,
                                          problem->chain[chainID].currentSolution->motion,
                                          &problem->chain[chainID].current2DProjectionTransform
                                         )
        )
      {
        bvh_removeTranslationFromTransform(
                                            problem->mc,
                                            &problem->chain[chainID].current2DProjectionTransform
                                          );

      if (
          (bvh_projectTo2D(problem->mc,&problem->chain[chainID].current2DProjectionTransform,problem->renderer,0,0)) &&
          (bvh_projectTo2D(problem->mc,problem->bvhTarget2DProjectionTransform,problem->renderer,0,0))
         )
      {
       for (unsigned int partID=0; partID<problem->chain[chainID].numberOfParts; partID++)
       {
         unsigned int jID=problem->chain[chainID].part[partID].jID;
         float thisSquared2DDistance=getSquared2DPointDistance(
                                                                (float) problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[0],
                                                                (float) problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[1],
                                                                (float) problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[0],
                                                                (float) problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[1]
                                                               );
         fprintf(stderr,"%0.2f,%0.2f -> %0.2f,%0.2f : ",
         problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[0],
         problem->chain[chainID].current2DProjectionTransform.joint[jID].pos2D[1],
         problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[0],
         problem->bvhTarget2DProjectionTransform->joint[jID].pos2D[1]
         );
         fprintf(stderr,"Joint squared %s distance is %0.2f\n",problem->mc->jointHierarchy[jID].jointName,thisSquared2DDistance);
         loss+=thisSquared2DDistance;
         ++numberOfSamples;
       }
     }

      } //Have a valid 2D transform
 } //Have a valid chain

 //I have left 0/0 on purpose to cause NaNs when projection errors occur
  loss = (float) loss/numberOfSamples;
  fprintf(stderr,"loss %0.2f\n",loss);
  return loss;
}


float iterateChainLoss(
                         struct ikProblem * problem,
                         unsigned int chainID
                        )
{
 unsigned int mIDS[3];
 mIDS[0]= problem->chain[chainID].part[0].mIDStart;
 mIDS[1]= problem->chain[chainID].part[0].mIDStart+1;
 mIDS[2]= problem->chain[chainID].part[0].mIDStart+2;
 float originalValues[3]={0};
 float delta[3]={0};

 originalValues[0] = problem->chain[chainID].currentSolution->motion[mIDS[0]];
 originalValues[1] = problem->chain[chainID].currentSolution->motion[mIDS[1]];
 originalValues[2] = problem->chain[chainID].currentSolution->motion[mIDS[2]];




}


int bruteForceChange(
                     struct BVH_MotionCapture * mc,
                     struct simpleRenderer *renderer,
                     struct MotionBuffer * solution,
                     float * averageError,
                     unsigned int fromElement,
                     unsigned int toElement,
                     unsigned int budget,
                     struct BVH_Transform * bvhSourceTransform,
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


int gatherBVH2DDistaces(
                        double * result,
                        unsigned int resultSize,
                        struct BVH_MotionCapture * mc,
                        struct simpleRenderer *renderer,
                        int useAllJoints,
                        BVHMotionChannelID onlyConsiderChildrenOfThisJoint,
                        struct BVH_Transform * bvhSourceTransform,
                        struct BVH_Transform * bvhTargetTransform
                       )
{
   unsigned int numberOfSamples=0;

   if (
        (bvh_projectTo2D(mc,bvhSourceTransform,renderer,0,0)) &&
        (bvh_projectTo2D(mc,bvhTargetTransform,renderer,0,0))
      )
      {
       //-----------------
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
                float thisSquared2DDistance=getSquared2DPointDistance(
                                                                      (float) bvhSourceTransform->joint[jID].pos2D[0],
                                                                      (float) bvhSourceTransform->joint[jID].pos2D[1],
                                                                      (float) bvhTargetTransform->joint[jID].pos2D[0],
                                                                      (float) bvhTargetTransform->joint[jID].pos2D[1]
                                                                     );

               if (numberOfSamples<resultSize)
               {
                 result[numberOfSamples]= (double) thisSquared2DDistance;
               } else
               {
                 fprintf(stderr,"gatherBVH2DDistaces: overflow..\n");
                 return 0;
               }

               numberOfSamples+=1;
              }
            }

     } //-----------------

 return  (numberOfSamples>0);
}







float approximateTargetFromMotionBuffer(
                                         struct BVH_MotionCapture * mc,
                                         struct simpleRenderer *renderer,
                                         struct MotionBuffer * solution,
                                         float * averageError,
                                         struct BVH_Transform * bvhTargetTransform
                                        )
{

 struct ikProblem problem={0};

 prepareProblem(
                 &problem,
                 mc,
                 renderer,
                 solution,
                 averageError,
                 bvhTargetTransform
                );

  viewProblem(
              &problem
             );

  float loss;

  loss=calculateChainLoss( &problem, 0 );


 return loss;
}




//./BVHTester --from Motions/05_01.bvh --selectJoints 0 23 hip eye.r eye.l abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot toe1-2.r toe5-3.r toe1-2.l toe5-3.l --testIK 4 100

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

  struct MotionBuffer * solution = mallocNewSolutionBuffer(mc);

  if (  (solution!=0) )
  {
    if ( bvh_copyMotionFrameToMotionBuffer(mc,solution,fIDSource) )
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
                                                         solution,
                                                         0,
                                                         &bvhTargetTransform
                                                        );

        fprintf(stderr,"2D Distance is %0.2f\n",error2D);
        result=1;
      }
    }
    freeSolutionBuffer(solution);
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
