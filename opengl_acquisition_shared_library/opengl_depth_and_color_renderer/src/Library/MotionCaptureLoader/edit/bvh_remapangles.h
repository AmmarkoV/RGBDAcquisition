#ifndef BVH_INTERPOLATE_H_INCLUDED
#define BVH_INTERPOLATE_H_INCLUDED

#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

float bvh_RemapAngleCentered0(float angle, unsigned int constrainOrientation);

float bvh_constrainAngleCentered0(float angle,unsigned int flipOrientation);


#ifdef __cplusplus
}
#endif

#endif // BVH_INTERPOLATE_H_INCLUDED
