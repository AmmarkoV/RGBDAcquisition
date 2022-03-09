#ifndef BVH_INVERSEKINEMATICS_H_INCLUDED
#define BVH_INVERSEKINEMATICS_H_INCLUDED

#include <string.h>
#include "../bvh_loader.h"
#include "../export/bvh_export.h"

#include "../../../../../../tools/PThreadWorkerPool/pthreadWorkerPool.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define IK_VERSION 0.38

#define MAXIMUM_CHAINS 16
#define MAXIMUM_PARTS_OF_CHAIN 16
#define MAXIMUM_PROBLEM_DESCRIPTION 64

enum bvhIKSolutionStatus
{
  BVH_IK_UNINITIALIZED=0,
  BVH_IK_NOTSTARTED,
  BVH_IK_STARTED,
  BVH_IK_FINISHED_ITERATION,
  BVH_IK_FINISHED_EVERYTHING,
  //--------------------
  BVH_IK_STATES
};

struct ikChainParts
{
 BVHJointID jID;
 //--------------
 float jointImportance;
 //--------------
 BVHMotionChannelID mIDStart;
 BVHMotionChannelID mIDEnd;
 //--------------
 float minimumLimitMID[4];
 float maximumLimitMID[4];
 float mAE[4];
 //--------------
 unsigned char limits;
 unsigned char maeDeclared;
 unsigned char evaluated;
 unsigned char smallChanges;
 unsigned char bigChanges;
 unsigned char endEffector;
 //unsigned char ignoreJointOwnError; just set jointImportance to 0.0
};
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------

struct ikChain
{
  //Thread information
  // --------------------------------------------------------------------------
  unsigned char parallel;
  unsigned char currentIteration;
  unsigned char status; // enum bvhIKSolutionStatus
  unsigned char permissionToStart;
  unsigned char terminate;
  unsigned char threadIsSpawned;
  // --------------------------------------------------------------------------
  float initialError;
  float previousError;
  float currentError;
  // --------------------------------------------------------------------------
  unsigned int encounteredAdoptedBest;
  unsigned int encounteredNumberOfNaNsAtStart;
  unsigned int encounteredExplodingGradients;
  unsigned int encounteredWorseSolutionsThanPrevious;
  // --------------------------------------------------------------------------
  unsigned int jobID;
  unsigned int groupID;
  // --------------------------------------------------------------------------
  unsigned int numberOfParts;
  struct ikChainParts part[MAXIMUM_PARTS_OF_CHAIN+1];
  struct MotionBuffer * currentSolution;
  struct BVH_Transform current2DProjectionTransform;
};
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------


struct ikConfiguration
{
  float maximumAcceptableStartingLoss;
  float learningRate;
  unsigned int iterations;
  unsigned int epochs;
  unsigned int considerPreviousSolution;
  unsigned int tryMaintainingLocalOptima;
  float spring;
  float gradientExplosionThreshold;
  unsigned int dumpScreenshots;
  unsigned int verbose;
  char dontUseSolutionHistory;
  float ikVersion;
};

static void printIkConfiguration(struct ikConfiguration * ikConfig)
{
  fprintf(stderr,"ikConfig ---------------\n");
  if (ikConfig!=0)
  {
   fprintf(stderr,"learningRate = %f \n",ikConfig->learningRate);
   fprintf(stderr,"maximumAcceptableStartingLoss = %f \n",ikConfig->maximumAcceptableStartingLoss);
   fprintf(stderr,"iterations = %u \n",ikConfig->iterations);
   fprintf(stderr,"epochs = %u \n",ikConfig->epochs);
   fprintf(stderr,"considerPreviousSolution = %u \n",ikConfig->considerPreviousSolution);
   fprintf(stderr,"tryMaintainingLocalOptima = %u \n",ikConfig->tryMaintainingLocalOptima);
   fprintf(stderr,"spring = %f \n",ikConfig->spring);
   fprintf(stderr,"gradientExplosionThreshold = %f \n",ikConfig->gradientExplosionThreshold);
   fprintf(stderr,"dumpScreenshots = %u \n",ikConfig->dumpScreenshots);
   fprintf(stderr,"verbose = %u \n",ikConfig->verbose);
   fprintf(stderr,"dontUseSolutionHistory = %u \n",(unsigned int) ikConfig->dontUseSolutionHistory);
   fprintf(stderr,"ikVersion = %f ( bin %f ) \n",ikConfig->ikVersion,IK_VERSION);
  }
  fprintf(stderr,"------------------------\n");
}

struct passIKContextToThread
{
    struct ikProblem * problem;
    struct ikConfiguration * ikConfig;
    unsigned int chainID;
    unsigned int threadID;
};


struct ikProblem
{
 //BVH file that reflects our problem
 struct BVH_MotionCapture * mc;
 //Renderer that handles 3D projections
 struct simpleRenderer *renderer;

 //Previous solution
 struct MotionBuffer * previousSolution;
 //Initial solution
 struct MotionBuffer * initialSolution;
 //Current solution
 struct MotionBuffer * currentSolution;

 //2D Projections Targeted
 struct BVH_Transform * bvhTarget2DProjectionTransform;

 //Chain of subproblems that need to be solved
 unsigned int numberOfChains;
 //unsigned int numberOfGroups; groups not needed since multiple problems are used, one for each "group"
 unsigned int numberOfJobs;

 struct ikChain chain[MAXIMUM_CHAINS+1];

 //We need a cutoff plane to prevent inverted
 //hierarchies from being compared..
 //----------------------------------------
 float nearCutoffPlane;
 char  nearCutoffPlaneDeclared;
 //----------------------------------------

 //Thread storage..
 //----------------------------------------
 struct workerPool threadPool;
 //----------------------------------------
 struct passIKContextToThread workerContext[MAXIMUM_CHAINS+1];

 //----------------------------------------
 char problemDescription[MAXIMUM_PROBLEM_DESCRIPTION+1];
};
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------


static struct ikProblem * allocateEmptyIKProblem()
{
    struct ikProblem * emptyProblem = (struct ikProblem * ) malloc(sizeof(struct ikProblem));
     if (emptyProblem!=0)
         {
            memset(emptyProblem,0,sizeof(struct ikProblem));
         } else
         { fprintf(stderr,"Failed to allocate memory for our IK Problem..\n"); }
     return emptyProblem;
}


int cleanProblem(struct ikProblem * problem);
int viewProblem(struct ikProblem * problem);

unsigned long GetTickCountMicrosecondsIK();

//Temporary call that allows outside control..
int approximateBodyFromMotionBufferUsingInverseKinematics(
                                         struct BVH_MotionCapture * mc,
                                         struct simpleRenderer *renderer,
                                         struct ikProblem * problem,
                                         struct ikConfiguration * ikConfig,
                                         //---------------------------------
                                         struct MotionBuffer * penultimateSolution,
                                         struct MotionBuffer * previousSolution,
                                         struct MotionBuffer * solution,
                                         struct MotionBuffer * groundTruth,
                                         //---------------------------------
                                         struct BVH_Transform * bvhTargetTransform,
                                         //---------------------------------
                                         unsigned int useMultipleThreads,
                                         //---------------------------------
                                         float * initialMAEInPixels,
                                         float * finalMAEInPixels,
                                         float * initialMAEInMM,
                                         float * finalMAEInMM
                                        );



int bvh_MirrorJointsThroughIK(
                               struct BVH_MotionCapture * mc,
                               const char * jointNameA,
                               const char * jointNameB
                             );

#ifdef __cplusplus
}
#endif

#endif
