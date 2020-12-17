#ifndef BVH_MEASURE_H_INCLUDED
#define BVH_MEASURE_H_INCLUDED

#include "../bvh_loader.h"
#include "../ik/bvh_inverseKinematics.h"

void compareMotionBuffers(const char * msg,struct MotionBuffer * guess,struct MotionBuffer * groundTruth);

void compareTwoMotionBuffers(struct BVH_MotionCapture * mc,const char * msg,struct MotionBuffer * guessA,struct MotionBuffer * guessB,struct MotionBuffer * groundTruth);

int bvhMeasureIterationInfluence(
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


int extractMinimaMaximaFromBVHList(const char * filename);

#endif
