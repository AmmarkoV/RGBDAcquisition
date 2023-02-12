#ifndef BVH_REMAPANGLES_H_INCLUDED
#define BVH_REMAPANGLES_H_INCLUDED

#include "../bvh_loader.h"

#include "../calculate/bvh_project.h"
#include "../../../../../../tools/AmMatrix/simpleRenderer.h"

#ifdef __cplusplus
extern "C"
{
#endif

float bvh_RemapAngleCentered0(float angle, unsigned int constrainOrientation);

float bvh_constrainAngleCentered0(float angle,unsigned int flipOrientation);
float bvh_normalizeAngle(float angle);


int bvh_normalizeRotations(struct BVH_MotionCapture * bvh);

int bvh_swapJointRotationAxis(struct BVH_MotionCapture * bvh,char inputRotationOrder,char swappedRotationOrder);

int bvh_swapJointNameRotationAxis(struct BVH_MotionCapture * bvh,const char * jointName,char inputRotationOrder,char swappedRotationOrder);


int dumpBVHAsProbabilitiesHeader(
                                 struct BVH_MotionCapture * mc,
                                 const char * filename,
                                 float *rangeMinimum,
                                 float *rangeMaximum,
                                 float *resolution
                                );


int countBodyDoF(struct BVH_MotionCapture * mc);

int countNumberOfHeatmapResolutions(
                                    struct BVH_MotionCapture * mc,
                                    float *rangeMinimum,
                                    float *rangeMaximum,
                                    float *resolution
                                   );

int dumpBVHAsProbabilitiesBody(
                                 struct BVH_MotionCapture * mc,
                                 const char * filename,
                                 struct simpleRenderer * renderer,
                                 BVHFrameID fID,
                                 int numberOfHeatmapTasks,
                                 int heatmapResolution,
                                 float *rangeMinimum,
                                 float *rangeMaximum,
                                 float *resolution
                             );


int bvh_studyMID2DImpact(
                           struct BVH_MotionCapture * bvh,
                           struct BVH_RendererConfiguration* renderingConfiguration,
                           BVHFrameID fID,
                           BVHMotionChannelID mIDRelativeToOneFrame,
                           float *rangeMinimum,
                           float *rangeMaximum,
                           float *resolution
                          );

int bvh_study3DJoint2DImpact(
                           struct BVH_MotionCapture * bvh,
                           struct BVH_RendererConfiguration* renderingConfiguration,
                           BVHFrameID fID,
                           BVHJointID jID,
                           float *rangeMinimum,
                           float *rangeMaximum,
                           float *resolution
                          );

#ifdef __cplusplus
}
#endif

#endif // BVH_REMAPANGLES_H_INCLUDED
