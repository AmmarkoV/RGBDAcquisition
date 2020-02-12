#ifndef BVH_EXPORT_H_INCLUDED
#define BVH_EXPORT_H_INCLUDED


#include "../bvh_loader.h"
#include "../bvh_transform.h"
#include "../bvh_project.h"

#ifdef __cplusplus
extern "C"
{
#endif

extern unsigned int filteredOutCSVBehindPoses;
extern unsigned int filteredOutCSVOutPoses;

int performPointProjectionsForFrameForcingPositionAndRotation(
                                                              struct BVH_MotionCapture * mc,
                                                              struct BVH_Transform * bvhTransform,
                                                              unsigned int fID,
                                                              struct simpleRenderer * renderer,
                                                              float * forcePosition,
                                                              float * forceRotation,
                                                              unsigned int occlusions,
                                                              unsigned int directRendering
                                                             );

int performPointProjectionsForFrame(
                                    struct BVH_MotionCapture * mc,
                                    struct BVH_Transform * bvhTransform,
                                    unsigned int fID,
                                    struct simpleRenderer * renderer,
                                    unsigned int occlusions,
                                    unsigned int directRendering
                                   );

int performPointProjectionsForMotionBuffer(
                                            struct BVH_MotionCapture * mc,
                                            struct BVH_Transform * bvhTransform,
                                            float * motionBuffer,
                                            struct simpleRenderer * renderer,
                                            unsigned int occlusions,
                                            unsigned int directRendering
                                           );

int dumpBVHToSVGCSV(
                    const char * directory ,
                    const char * filename,
                    int convertToSVG,
                    int convertToCSV,int useCSV_2D_Output,int useCSV_3D_Output,int useCSV_BVH_Output,
                    struct BVH_MotionCapture * mc,
                    unsigned int csvOrientation,
                    struct BVH_RendererConfiguration * renderConfig,
                    unsigned int occlusions,
                    unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                    unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                    unsigned int filterWeirdSkeletons,
                    unsigned int encodeRotationsAsRadians
                   );

#ifdef __cplusplus
}
#endif

#endif // BVH_EXPORT_H_INCLUDED
