#ifndef BVH_INVERSEKINEMATICS_H_INCLUDED
#define BVH_INVERSEKINEMATICS_H_INCLUDED


#include "../bvh_loader.h"

#include "../export/bvh_export.h"

#ifdef __cplusplus
extern "C"
{
#endif


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

int BVHTestIK(
              struct BVH_MotionCapture * mc,
              unsigned int iterations,
              unsigned int epochs,
              unsigned int fIDSource,
              unsigned int fIDTarget
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
