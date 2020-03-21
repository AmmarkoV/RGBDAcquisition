#ifndef BVH_TO_TRAJECTORYPARSERTRI_H_INCLUDED
#define BVH_TO_TRAJECTORYPARSERTRI_H_INCLUDED

#include "../bvh_loader.h"
#include "../calculate/bvh_to_tri_pose.h"

#ifdef __cplusplus
extern "C"
{
#endif

int dumpBVHToTrajectoryParserTRI(
                                  const char * filename ,
                                  struct BVH_MotionCapture * mc,
                                  struct bvhToTRI * bvhtri ,
                                  unsigned int usePosition,
                                  unsigned int includeSpheres
                                );

#ifdef __cplusplus
}
#endif

#endif // BVH_TO_TRAJECTORYPARSER_H_INCLUDED
