#ifndef BVH_LOADER_H_INCLUDED
#define BVH_LOADER_H_INCLUDED

#define MAX_BVH_JOINT_NAME 128

enum CHANNEL_NAMES
{
  BVH_POSITION_NONE=0,
  BVH_POSITION_X,
  BVH_POSITION_Y,
  BVH_POSITION_Z,
  BVH_ROTATION_X,
  BVH_ROTATION_Y,
  BVH_ROTATION_Z,
  //--------------------
  BVH_VALID_CHANNEL_NAMES
};


struct BVH_Joint
{
  //--------------------
  char jointName[MAX_BVH_JOINT_NAME];
  unsigned int parentJoint;

  //--------------------
  float offset[3];

  //--------------------
  float channels[7];
  char  channelType[7];
  char  loadedChannels;

  //--------------------
  char isRoot;

  //--------------------
  char hasEndSite;
  float endSiteOffset[3];
};


struct BVH_MotionCapture
{
  //Header
  unsigned int numberOfFrames;
  unsigned int numberOfFramesEncountered;
  float frameTime;


  //Joint Hierarchy
  unsigned int MAX_jointHierarchySize;
  unsigned int jointHierarchySize;
  struct BVH_Joint jointHierarchy[128];


  //Motion
  unsigned int numberOfValuesPerFrame;
  unsigned int motionValuesSize;
  float * motionValues;

  //Internal Variables..

};


int loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion);

#endif // BVH_LOADER_H_INCLUDED
