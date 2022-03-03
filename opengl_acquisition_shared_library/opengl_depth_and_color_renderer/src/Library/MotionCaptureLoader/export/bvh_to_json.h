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
                        int wiped2DOutput,
                        int wiped3DOutput,
                        int wipedBVHOutput,
                        int * did2DOutputPreExist,
                        int * did3DOutputPreExist,
                        int * didBVHOutputPreExist,
                        const char * filenameInput,
                        const char * filename3D,
                        const char * filenameBVH,
                        float fx,
                        float fy,
                        float cx,
                        float cy,
                        float near,
                        float far,
                        float width,
                        float height
                       );

int dumpBVHToJSONFooter(
                        struct BVH_MotionCapture * mc,
                        const char * filenameInput,
                        const char * filename3D,
                        const char * filenameBVH
                       );

int dumpBVHToJSONBody(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform * bvhTransform,
                       struct simpleRenderer * renderer,
                       unsigned int fID,
                       const char * filenameInput,
                       const char * filename3D,
                       const char * filenameBVH,
                       int didInputOutputPreExist,
                       int did3DOutputPreExist,
                       int didBVHOutputPreExist,
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
