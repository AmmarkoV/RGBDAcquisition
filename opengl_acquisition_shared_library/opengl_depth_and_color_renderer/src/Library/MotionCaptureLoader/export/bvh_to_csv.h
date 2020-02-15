#ifndef BVH_TO_CSV_H_INCLUDED
#define BVH_TO_CSV_H_INCLUDED


#include "../bvh_loader.h"
#include "../bvh_transform.h"
#include "../../../../../../tools/AmMatrix/simpleRenderer.h"
#include "bvh_export.h"


#ifdef __cplusplus
extern "C"
{
#endif



int dumpBVHToCSVHeader(
                       struct BVH_MotionCapture * mc,
                       const char * filename2D,
                       const char * filename3D,
                       const char * filenameBVH
                      );


int dumpBVHToCSVBody(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform * bvhTransform,
                       struct simpleRenderer * renderer,
                       unsigned int fID,
                       const char * filename2D,
                       const char * filename3D,
                       const char * filenameBVH,
                       unsigned int csvOrientation,
                       struct filteringResults * filterStats,
                       unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                       unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                       unsigned int filterWeirdSkeletons,
                       unsigned int encodeRotationsAsRadians
                      );


#ifdef __cplusplus
}
#endif

#endif // BVH_TO_SVG_H_INCLUDED
