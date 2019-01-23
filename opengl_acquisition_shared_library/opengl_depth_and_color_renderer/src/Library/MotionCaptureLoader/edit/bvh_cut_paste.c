#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "bvh_cut_paste.h"


int scanJointHierarchyUntilThisGroupEnds(
                                         struct BVH_MotionCapture * mc,
                                         BVHJointID jID,
                                         BVHJointID * jIDLastGroupJoint,
                                         unsigned int * numberOfChannelsContained
                                        )
{
  fprintf(stderr,"scanJointHierarchyUntilThisGroupEnds(%s): contains ",mc->jointHierarchy[jID].jointName);

  *jIDLastGroupJoint=jID;
  unsigned int targetHierarchyLevel=mc->jointHierarchy[jID].hierarchyLevel;
  *numberOfChannelsContained = mc->jointHierarchy[jID].loadedChannels;

  ++jID;//Start from the next joint..
  while (jID<mc->numberOfFrames)
  {
    if (targetHierarchyLevel>=mc->jointHierarchy[jID].hierarchyLevel)
    {
      fprintf(stderr,"all of them are %u \n",*numberOfChannelsContained);
      //We have reached the end..!
      *jIDLastGroupJoint=jID;
      return 1;
    } else
    {
      *numberOfChannelsContained+=mc->jointHierarchy[jID].loadedChannels;
      fprintf(stderr,"(%s-%u) ",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].loadedChannels);
      jID++;
    }
  }

 fprintf(stderr,"\n");
 return 0;
}



int checkIfJointsHaveSameGraphOutline(
                                       struct BVH_MotionCapture * mc,
                                       BVHJointID jIDA,
                                       BVHJointID jIDB,
                                       unsigned int * rangeOfJIDA,
                                       unsigned int * rangeOfJIDB,
                                       unsigned int * numberOfChannelsContainedJIDA,
                                       unsigned int * numberOfChannelsContainedJIDB
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
                                              &jIDALastJoint,
                                              numberOfChannelsContainedJIDA
                                            )
      ) &&
      (
        scanJointHierarchyUntilThisGroupEnds(
                                              mc,
                                              jIDB,
                                              &jIDBLastJoint,
                                              numberOfChannelsContainedJIDB
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
       return ((rA==rB) && (*numberOfChannelsContainedJIDA==*numberOfChannelsContainedJIDB));
     }

  return 0;
}


float * allocateBufferThatCanContainJointAndChildren(
                                                      struct BVH_MotionCapture * mc,
                                                      BVHJointID jID
                                                    )
{
  unsigned int numberOfChannels=0;
  BVHJointID jIDLastJoint;
  if  (
        scanJointHierarchyUntilThisGroupEnds(
                                              mc,
                                              jID,
                                              &jIDLastJoint,
                                              &numberOfChannels
                                            )
      )
      {
        return (float *) malloc(sizeof(float) * (numberOfChannels));
      }
  return 0;
}



int copyJointAndChildrenToBuffer(
                                 struct BVH_MotionCapture * mc,
                                 float * buffer,
                                 BVHJointID jID,
                                 unsigned int rangeNumber,
                                 BVHFrameID  fID
                                )
{
  unsigned int mID = (fID * mc->numberOfValuesPerFrame) + mc->jointToMotionLookup[jID].jointMotionOffset;
  memcpy(
         buffer ,
         &mc->motionValues[mID],
         rangeNumber * sizeof(float)
        );

 return 1;
}

int copyBufferToJointAndChildren(
                                 struct BVH_MotionCapture * mc,
                                 float * buffer,
                                 BVHJointID jID,
                                 unsigned int rangeNumber,
                                 BVHFrameID  fID
                                )
{
  unsigned int mID = (fID * mc->numberOfValuesPerFrame) + mc->jointToMotionLookup[jID].jointMotionOffset;
  memcpy(
         &mc->motionValues[mID],
         buffer ,
         rangeNumber * sizeof(float)
        );

 return 1;
}
