#ifndef BVH_HARDCODEDPROBLEMS_H_INCLUDED
#define BVH_HARDCODEDPROBLEMS_H_INCLUDED

#include "bvh_inverseKinematics.h"

#ifdef __cplusplus
extern "C"
{
#endif

int prepareDefaultFaceProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform,
    int standalone
);

int prepareDefaultRightHandProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform,
    int standalone
);

int prepareDefaultLeftHandProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform,
    int standalone
);

int prepareDefaultBodyProblem(
    struct ikProblem * problem,
    struct BVH_MotionCapture * mc,
    struct simpleRenderer *renderer,
    struct MotionBuffer * previousSolution,
    struct MotionBuffer * solution,
    struct BVH_Transform * bvhTargetTransform
);

float convertStartEndTimeFromMicrosecondsToFPSIK(unsigned long startTime, unsigned long endTime);

int bvhTestIK(
              struct BVH_MotionCapture * mc,
              float lr,
              float spring,
              unsigned int iterations,
              unsigned int epochs,
              unsigned int fIDPrevious,
              unsigned int fIDSource,
              unsigned int fIDTarget,
              unsigned int multiThreaded
             );



#ifdef __cplusplus
}
#endif

#endif
