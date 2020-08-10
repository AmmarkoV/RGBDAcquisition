#ifndef BVH_FILTER_H_INCLUDED
#define BVH_FILTER_H_INCLUDED


#include "../bvh_loader.h"

#include "../export/bvh_export.h"

#ifdef __cplusplus
extern "C"
{
#endif

int filterOutPosesThatAreGimbalLocked(struct BVH_MotionCapture * mc,float threshold);

int filterOutPosesThatAreCloseToRules(struct BVH_MotionCapture * bvhMotion,int argc,const char **argv);


int probeForFilterRules(struct BVH_MotionCapture * mc,int argc,const char **argv);

#ifdef __cplusplus
}
#endif

#endif
