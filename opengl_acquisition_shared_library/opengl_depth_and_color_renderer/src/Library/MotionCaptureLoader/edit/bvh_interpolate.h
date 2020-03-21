#ifndef BVH_INTERPOLATE_H_INCLUDED
#define BVH_INTERPOLATE_H_INCLUDED

#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

int bvh_InterpolateMotion(struct BVH_MotionCapture * mc);

#ifdef __cplusplus
}
#endif

#endif // BVH_INTERPOLATE_H_INCLUDED
