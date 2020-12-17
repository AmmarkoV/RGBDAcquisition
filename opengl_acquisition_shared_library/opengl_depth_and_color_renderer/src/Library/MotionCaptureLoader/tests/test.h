#ifndef BVH_TEST_H_INCLUDED
#define BVH_TEST_H_INCLUDED

#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

void testPrintout(struct BVH_MotionCapture * bvhMotion,const char * jointName);

#ifdef __cplusplus
}
#endif

#endif // BVH_TEST_H_INCLUDED
