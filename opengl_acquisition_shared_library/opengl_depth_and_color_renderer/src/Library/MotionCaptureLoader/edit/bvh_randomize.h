#ifndef BVH_RANDOMIZE_H_INCLUDED
#define BVH_RANDOMIZE_H_INCLUDED


#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif


float randomFloatA( float minVal, float maxVal );


int bvh_RandomizeBasedOnIKProblem(
                                  struct BVH_MotionCapture * mc,
                                  const char * ikProblemName
                                 );

int bvh_PerturbJointAnglesRange(
                           struct BVH_MotionCapture * mc,
                           unsigned int numberOfValues,
                           float  start,
                           float  end,
                           unsigned int specificChannel,
                           const char **argv,
                           unsigned int iplus2
                          );


int bvh_PerturbJointAngles(
                           struct BVH_MotionCapture * mc,
                           unsigned int numberOfValues,
                           float  deviation,
                           const char **argv,
                           unsigned int iplus2
                          );


int bvh_RandomizeSingleMIDInRange(
                                  struct BVH_MotionCapture * mc,
                                  BVHMotionChannelID mID,
                                  float start,
                                  float end
                                 );

int bvh_eraseJoints(
                    struct BVH_MotionCapture * mc,
                    unsigned int numberOfValues,
                    unsigned int includeEndSites,
                    const char **argv,
                    unsigned int iplus1
                   );

int bvh_RandomizePositionRotation(
                                  struct BVH_MotionCapture * mc,
                                  float * minimumPosition,
                                  float * minimumRotation,
                                  float * maximumPosition,
                                  float * maximumRotation
                                 );


int bvh_RandomizePositionRotation2Ranges(
                                         struct BVH_MotionCapture * mc,
                                         float * minimumPositionRangeA,
                                         float * minimumRotationRangeA,
                                         float * maximumPositionRangeA,
                                         float * maximumRotationRangeA,
                                         float * minimumPositionRangeB,
                                         float * minimumRotationRangeB,
                                         float * maximumPositionRangeB,
                                         float * maximumRotationRangeB
                                        );


int bvh_RandomizePositionFrom2DRotation2Ranges(
                                               struct BVH_MotionCapture * mc,
                                               float * minimumRotationRangeA,
                                               float * maximumRotationRangeA,
                                               float * minimumRotationRangeB,
                                               float * maximumRotationRangeB,
                                               float minimumDepth,float maximumDepth,
                                               float fX,float fY,float cX,float cY,unsigned int width,unsigned int height
                                              );

int bvh_RandomizePositionFrom2D(
                                 struct BVH_MotionCapture * mc,
                                 float * minimumRotation,
                                 float * maximumRotation,
                                 float minimumDepth,float maximumDepth,
                                 float fX,float fY,float cX,float cY,unsigned int width,unsigned int height
                                );

/*
int bvh_TestRandomizationLimitsXYZ(
                                   struct BVH_MotionCapture * mc,
                                   float * minimumPosition,
                                   float * maximumPosition
                                  );
*/


#ifdef __cplusplus
}
#endif

#endif
