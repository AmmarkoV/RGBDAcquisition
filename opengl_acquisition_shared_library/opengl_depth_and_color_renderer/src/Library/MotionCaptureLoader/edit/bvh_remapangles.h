#ifndef BVH_INTERPOLATE_H_INCLUDED
#define BVH_INTERPOLATE_H_INCLUDED

#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

double bvh_RemapAngleCentered0(double angle, unsigned int constrainOrientation);

double bvh_constrainAngleCentered0(double angle,unsigned int flipOrientation);


#ifdef __cplusplus
}
#endif

#endif // BVH_INTERPOLATE_H_INCLUDED
