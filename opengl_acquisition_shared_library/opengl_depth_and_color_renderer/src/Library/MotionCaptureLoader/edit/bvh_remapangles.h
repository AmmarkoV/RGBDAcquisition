#ifndef BVH_INTERPOLATE_H_INCLUDED
#define BVH_INTERPOLATE_H_INCLUDED

#include "../bvh_loader.h"

#include "../calculate/bvh_project.h"
#include "../../../../../../tools/AmMatrix/simpleRenderer.h"

#ifdef __cplusplus
extern "C"
{
#endif

float bvh_RemapAngleCentered0(float angle, unsigned int constrainOrientation);

float bvh_constrainAngleCentered0(float angle,unsigned int flipOrientation);

int bvh_swapJointRotationAxis(struct BVH_MotionCapture * bvh,char inputRotationOrder,char swappedRotationOrder);

int bvh_swapJointNameRotationAxis(struct BVH_MotionCapture * bvh,const char * jointName,char inputRotationOrder,char swappedRotationOrder);

int bvh_studyMID2DImpact(
                           struct BVH_MotionCapture * bvh,
                           struct BVH_RendererConfiguration* renderingConfiguration,
                           BVHFrameID fID,
                           BVHMotionChannelID mIDRelativeToOneFrame,
                           float rangeMinimum,
                           float rangeMaximum
                        );

#ifdef __cplusplus
}
#endif

#endif // BVH_INTERPOLATE_H_INCLUDED
