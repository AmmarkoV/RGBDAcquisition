#ifndef BVH_TO_JSON_H_INCLUDED
#define BVH_TO_JSON_H_INCLUDED


#include "../bvh_loader.h"
#include "../calculate/bvh_transform.h"
#include "../mathLibrary.h"

 #include "bvh_export.h"


#ifdef __cplusplus
extern "C"
{
#endif

int dumpBVHToJSONHeader(
                        struct BVH_MotionCapture * mc,
                        const char * filenameInput,
                        const char * filenameBVH
                       );

int dumpBVHToJSONFooter(
                        struct BVH_MotionCapture * mc,
                        const char * filenameInput,
                        const char * filenameBVH
                       );

int dumpBVHToJSONBody(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform * bvhTransform,
                       struct simpleRenderer * renderer,
                       unsigned int fID,
                       const char * filenameInput,
                       const char * filenameBVH,
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
