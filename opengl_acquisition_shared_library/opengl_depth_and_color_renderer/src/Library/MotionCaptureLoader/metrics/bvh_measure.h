#ifndef BVH_MEASURE_H_INCLUDED
#define BVH_MEASURE_H_INCLUDED

#include "../bvh_loader.h"
#include "../ik/bvh_inverseKinematics.h"

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

#endif
