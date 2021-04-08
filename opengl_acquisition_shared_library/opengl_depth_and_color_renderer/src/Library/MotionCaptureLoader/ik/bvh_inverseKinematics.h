#ifndef BVH_INVERSEKINEMATICS_H_INCLUDED
#define BVH_INVERSEKINEMATICS_H_INCLUDED


#include "../bvh_loader.h"
#include "../export/bvh_export.h"

#include "../../../../../../tools/PThreadWorkerPool/pthreadWorkerPool.h"

#ifdef __cplusplus
extern "C"
{
#endif

#define IK_VERSION 0.25

#define MAXIMUM_CHAINS 10
#define MAXIMUM_PARTS_OF_CHAIN 10

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
 char limits;
 char maeDeclared;
 char evaluated;
 char dontTrustInitialSolution;
 char smallChanges;
 char bigChanges;
 char endEffector;
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
  struct ikChainParts part[MAXIMUM_PARTS_OF_CHAIN];
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
  float ikVersion;
};



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
 unsigned int numberOfGroups;
 unsigned int numberOfJobs;

 struct ikChain chain[MAXIMUM_CHAINS];

 //Thread storage..
 //----------------------------------------
 struct workerPool threadPool;
 //----------------------------------------
 struct passIKContextToThread workerContext[MAXIMUM_CHAINS];

 //----------------------------------------
 char problemDescription[64];   
};
//---------------------------------------------------------
//---------------------------------------------------------
//---------------------------------------------------------




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
