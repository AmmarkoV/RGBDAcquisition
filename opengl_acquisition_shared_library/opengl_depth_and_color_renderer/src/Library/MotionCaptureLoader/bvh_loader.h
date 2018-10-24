/** @file bvh_loader.h
 *  @brief  BVH file parser
            part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
            This is a very clean-cut parser of bvh file that will load a struct BVH_MotionCapture using the loadBVH command
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef BVH_LOADER_H_INCLUDED
#define BVH_LOADER_H_INCLUDED

/**
* @brief MAX_BVH_JOINT_NAME is the maximum label size for Joint names
* @ingroup BVH
*/
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


/**
* @brief Each BVH Joint has a number of properties,offsets and channels that are modeled using this structure
* @ingroup BVH
*/
struct BVH_Joint
{
  //--------------------
  char isRoot;
  char hasBrace;
  char isEndSite;
  char hasEndSite;
  //--------------------
  char jointName[MAX_BVH_JOINT_NAME];
  //--------------------
  unsigned int parentJoint;
  unsigned int endSiteJoint;
  unsigned int hierarchyLevel;
  //--------------------
  float offset[3];
  //--------------------
  float channels[7];
  char  channelType[7];
  char  loadedChannels;
  //--------------------
};


/**
* @brief Each BVH Motion Capture file has a hierarchy of BVH_Joints and an array of motion values that should be accessed using
         the helper functions
* @ingroup BVH
*/
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


/**
* @brief Load a BVH file by giving a filename and filling in a BVH_MotionCapture struct
* @ingroup BVH
* @param  C-String with path to BVH File
* @param  pointer to an allocated BVH_MOtionCapture struct
*/
int loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion);

#endif // BVH_LOADER_H_INCLUDED
