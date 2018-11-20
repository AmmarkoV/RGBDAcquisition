#ifndef BVH_EXPORT_H_INCLUDED
#define BVH_EXPORT_H_INCLUDED


#include "../bvh_loader.h"
#include "../bvh_transform.h"

extern unsigned int filteredOutCSVBehindPoses;
extern unsigned int filteredOutCSVOutPoses;


int dumpBVHToSVGCSV(
                    const char * directory ,
                    const char * filename,
                    int convertToSVG,
                    int convertToCSV,
                    struct BVH_MotionCapture * mc,
                    unsigned int width,
                    unsigned int height,
                    unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                    unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                    unsigned int filterWeirdSkeletons,
                    unsigned int encodeRotationsAsRadians
                   );


#endif // BVH_EXPORT_H_INCLUDED
