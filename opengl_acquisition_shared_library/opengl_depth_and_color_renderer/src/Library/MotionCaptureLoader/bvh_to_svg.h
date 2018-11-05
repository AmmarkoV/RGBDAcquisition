#ifndef BVH_TO_SVG_H_INCLUDED
#define BVH_TO_SVG_H_INCLUDED


#include "bvh_loader.h"
#include "bvh_transform.h"

int dumpBVHToSVG(
                 const char * directory ,
                 struct BVH_MotionCapture * mc,
                 unsigned int width,
                 unsigned int height,
                 float * positionOffset
                 );

#endif // BVH_TO_SVG_H_INCLUDED
