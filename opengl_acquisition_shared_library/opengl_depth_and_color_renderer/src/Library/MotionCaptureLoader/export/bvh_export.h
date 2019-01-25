#ifndef BVH_EXPORT_H_INCLUDED
#define BVH_EXPORT_H_INCLUDED


#include "../bvh_loader.h"
#include "../bvh_transform.h"
#include "../bvh_project.h"

extern unsigned int filteredOutCSVBehindPoses;
extern unsigned int filteredOutCSVOutPoses;



int performPointProjectionsForFrame(
                                    struct BVH_MotionCapture * mc,
                                    struct BVH_Transform * bvhTransform,
                                    unsigned int fID,
                                    struct simpleRenderer * renderer,
                                    unsigned int occlusions
                                   );

int performPointProjectionsForMotionBuffer(
                                            struct BVH_MotionCapture * mc,
                                            struct BVH_Transform * bvhTransform,
                                            float * motionBuffer,
                                            struct simpleRenderer * renderer,
                                            unsigned int occlusions
                                           );

int dumpBVHToSVGCSV(
                    const char * directory ,
                    const char * filename,
                    int convertToSVG,
                    int convertToCSV,
                    struct BVH_MotionCapture * mc,
                    unsigned int width,
                    unsigned int height,
                    float fX,
                    float fY,
                    unsigned int occlusions,
                    unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                    unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                    unsigned int filterWeirdSkeletons,
                    unsigned int encodeRotationsAsRadians
                   );


#endif // BVH_EXPORT_H_INCLUDED
