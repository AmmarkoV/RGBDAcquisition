#include <stdio.h>
#include <stdlib.h>
#include "bvh_cut_paste.h"


int scanJointHierarchyUntilThisGroupEnds(
                                         struct BVH_MotionCapture * mc,
                                         BVHJointID jID,
                                         BVHJointID * jIDLastGroupJoint
                                        )
{
  fprintf(stderr,"scanJointHierarchyUntilThisGroupEnds(%s): contains ",mc->jointHierarchy[jID].jointName);

  *jIDLastGroupJoint=jID;
  unsigned int targetHierarchyLevel=mc->jointHierarchy[jID].hierarchyLevel;
  while (jID<mc->numberOfFrames)
  {
    if (targetHierarchyLevel>=mc->jointHierarchy[jID].hierarchyLevel)
    {
       fprintf(stderr,"(%s) ",mc->jointHierarchy[jID].jointName);
      //We have reached the end..!
      *jIDLastGroupJoint=jID;
      return 1;
    } else
    {
      jID++;
    }
  }
 return 0;
}



int checkIfJointsHaveSameGraphOutline(
                                       struct BVH_MotionCapture * mc,
                                       BVHJointID jIDA,
                                       BVHJointID jIDB,
                                       unsigned int * rangeOfJIDA,
                                       unsigned int * rangeOfJIDB
                                     )
{
  //We assume jIDA is before jIDB if now we fix it
  if (jIDA>jIDB)
  {
    BVHJointID tmp=jIDA;
    jIDA=jIDB;
    jIDB=tmp;
  }
  //----------------------------

  BVHJointID jIDALastJoint,jIDBLastJoint;

  //We see the range of each of the joint graphs
  if (
      (
        scanJointHierarchyUntilThisGroupEnds(
                                              mc,
                                              jIDA,
                                              &jIDALastJoint
                                            )
      ) &&
      (
        scanJointHierarchyUntilThisGroupEnds(
                                              mc,
                                              jIDB,
                                              &jIDBLastJoint
                                            )
      )
     )
     {
       unsigned int rA = (jIDALastJoint-jIDA);
       unsigned int rB = (jIDBLastJoint-jIDB);

       //If the last 2 arguments are not set to null we use them..!
       if (rangeOfJIDA!=0) { *rangeOfJIDA = rA; }
       if (rangeOfJIDB!=0) { *rangeOfJIDB = rB; }

       //If they are the same we can copy paste one to the other
       return (rA==rB);
     }

  return 0;
}


float * allocateBufferThatCanContainJointAndChildren(
                                                      struct BVH_MotionCapture * mc,
                                                      BVHJointID jID
                                                    )
{
  BVHJointID jIDLastJoint;
  if  (
        scanJointHierarchyUntilThisGroupEnds(
                                              mc,
                                              jID,
                                              &jIDLastJoint
                                            )
      )
      {
        float * buffer = (float *) malloc(sizeof(float) * (jIDLastJoint-jID));
        return buffer;
      }
  return 0;
}



int copyJointAndChildrenToBuffer(
                                 struct BVH_MotionCapture * mc,
                                 float * buffer,
                                 BVHJointID jID,
                                 unsigned int rangeNumber,
                                 unsigned int mID
                                )
{
  return 0;
}

int copyBufferToJointAndChildren(
                                 struct BVH_MotionCapture * mc,
                                 float * buffer,
                                 BVHJointID jID,
                                 unsigned int rangeNumber,
                                 unsigned int mID
                                )
{
  return 0;
}
