#include <stdio.h>
#include <stdlib.h>
#include "bvh_cut_paste.h"




int checkIfJointsHaveSameGraphOutline(
                                       struct BVH_MotionCapture * mc,
                                       BVHJointID jIDA,
                                       BVHJointID jIDB
                                     )
{
  return 0;
}


int allocateBufferThatCanContainJointAndChildren(
                                                  struct BVH_MotionCapture * mc,
                                                  BVHJointID jID
                                                 )
{
  return 0;
}



int copyJointAndChildrenToBuffer(
                                 struct BVH_MotionCapture * mc,
                                 BVHJointID jID
                                )
{
  return 0;
}

int copyBufferToJointAndChildren(
                                 struct BVH_MotionCapture * mc,
                                 BVHJointID jID
                                )
{
  return 0;
}
