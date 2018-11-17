#ifndef BVH_TO_BVH_H_INCLUDED
#define BVH_TO_BVH_H_INCLUDED


#include "../bvh_loader.h"
#include "../bvh_transform.h"
#include "../../../../../../tools/AmMatrix/simpleRenderer.h"

int dumpBVHToBVH(
                  const char * bvhFilename,
                  struct BVH_MotionCapture * mc
                );


#endif // BVH_TO_BVH_H_INCLUDED
