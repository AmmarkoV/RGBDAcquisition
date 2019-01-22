#ifndef BVH_CUT_PASTE_H_INCLUDED
#define BVH_CUT_PASTE_H_INCLUDED


#include "../bvh_loader.h"

int checkIfJointsHaveSameGraphOutline(
                                       struct BVH_MotionCapture * mc,
                                       BVHJointID jIDA,
                                       BVHJointID jIDB
                                     );

int allocateBufferThatCanContainJointAndChildren(
                                                  struct BVH_MotionCapture * mc,
                                                  BVHJointID jID
                                                 );

int copyJointAndChildrenToBuffer(
                                 struct BVH_MotionCapture * mc,
                                 BVHJointID jID
                                );

int copyBufferToJointAndChildren(
                                 struct BVH_MotionCapture * mc,
                                 BVHJointID jID
                                );
#endif
