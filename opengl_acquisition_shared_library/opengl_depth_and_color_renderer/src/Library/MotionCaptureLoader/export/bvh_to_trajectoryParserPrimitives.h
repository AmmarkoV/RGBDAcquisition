#ifndef BVH_TO_TRAJECTORYPARSERPRIMITIVES_H_INCLUDED
#define BVH_TO_TRAJECTORYPARSERPRIMITIVES_H_INCLUDED


#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

int dumpBVHToTrajectoryParserPrimitives(const char * filename , struct BVH_MotionCapture * mc);


#ifdef __cplusplus
}
#endif

#endif // BVH_TO_TRAJECTORYPARSERPRIMITIVES_H_INCLUDED
