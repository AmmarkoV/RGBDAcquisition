#ifndef BVH_EXPORT_H_INCLUDED
#define BVH_EXPORT_H_INCLUDED


#include "../bvh_loader.h"
#include "../bvh_transform.h"

int dumpBVHToSVGCSV(
                    const char * directory ,
                    int convertToSVG,
                    int convertToCSV,
                    struct BVH_MotionCapture * mc,
                    unsigned int width,
                    unsigned int height,
                    unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                    unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                    unsigned int encodeRotationsAsRadians
                   );


#endif // BVH_EXPORT_H_INCLUDED
