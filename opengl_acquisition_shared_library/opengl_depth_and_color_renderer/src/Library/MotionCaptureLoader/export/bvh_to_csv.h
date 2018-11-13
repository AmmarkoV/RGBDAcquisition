#ifndef BVH_TO_CSV_H_INCLUDED
#define BVH_TO_CSV_H_INCLUDED


#include "../bvh_loader.h"
#include "../bvh_transform.h"
#include "../../../../../../tools/AmMatrix/simpleRenderer.h"

extern unsigned int filteredOutCSVPoses;

int dumpBVHToCSVHeader(
                       struct BVH_MotionCapture * mc,
                       const char * filename
                      );


int dumpBVHToCSVBody(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform * bvhTransform,
                       struct simpleRenderer * renderer,
                       float * objectRotationOffset,
                       unsigned int fID,
                       const char * filename,
                       unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                       unsigned int encodeRotationsAsRadians
                      );

#endif // BVH_TO_SVG_H_INCLUDED
