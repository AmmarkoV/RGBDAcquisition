#ifndef BVH_TO_C_H_INCLUDED
#define BVH_TO_C_H_INCLUDED


#include "../bvh_loader.h"
#include "../calculate/bvh_transform.h"
#include "../mathLibrary.h"
 
#ifdef __cplusplus
extern "C"
{
#endif
 
  void bvh_print_C_Header(struct BVH_MotionCapture * bvhMotion);

#ifdef __cplusplus
}
#endif

#endif // BVH_TO_C_H_INCLUDED
