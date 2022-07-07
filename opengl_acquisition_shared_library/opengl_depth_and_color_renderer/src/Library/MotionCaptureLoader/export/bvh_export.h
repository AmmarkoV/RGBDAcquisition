#ifndef BVH_EXPORT_H_INCLUDED
#define BVH_EXPORT_H_INCLUDED


#include "../bvh_loader.h"
#include "../calculate/bvh_transform.h"
#include "../calculate/bvh_project.h"

#ifdef __cplusplus
extern "C"
{
#endif


struct filteringResults
{
 unsigned int invisibleJoints;
 unsigned int visibleJoints;
 unsigned int filteredOutCSVBehindPoses;
 unsigned int filteredOutCSVOutPoses;
 unsigned int filteredOutCSVPoses;
};

/*
extern unsigned int filteredOutCSVBehindPoses;
extern unsigned int filteredOutCSVOutPoses;
*/


int bvhExportFileExists(const char * filename);

int bvhExportFileWipe(const char * filename);

float bvhExportEulerAngleToRadiansIfNeeded( float eulerAngle , unsigned int isItNeeded);



int bvhExportSkeletonFilter(
                            struct BVH_MotionCapture * mc,
                            struct BVH_Transform * bvhTransform,
                            struct simpleRenderer * renderer,
                            struct filteringResults * filterStats,
                            unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                            unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                            unsigned int filterWeirdSkeletons
                           );


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


int dumpBVHTo_JSON_SVG_CSV(
                           const char * directory,
                           const char * filename,
                           int convertToJSON,
                           int convertToSVG,
                           int convertToCSV,
                           int convertToAngleHeatmap,
                           int useCSV_2D_Output,int useCSV_3D_Output,int useCSV_BVH_Output,
                           int wipe_2D_Output,int wipe_3D_Output,int wipe_BVH_Output,
                           struct BVH_MotionCapture * mc,
                           struct BVH_RendererConfiguration * renderConfig,
                           struct filteringResults * filterStats,
                           unsigned int sampleSkip,
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
