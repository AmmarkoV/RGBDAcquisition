#ifndef BVH_TO_BVH_H_INCLUDED
#define BVH_TO_BVH_H_INCLUDED


#include "../bvh_loader.h"
#include "../bvh_transform.h"
#include "../../../../../../tools/AmMatrix/simpleRenderer.h"

#ifdef __cplusplus
extern "C"
{
#endif

int dumpBVHToBVH(
                  const char * bvhFilename,
                  struct BVH_MotionCapture * mc
                );

#ifdef __cplusplus
}
#endif

#endif // BVH_TO_BVH_H_INCLUDED
