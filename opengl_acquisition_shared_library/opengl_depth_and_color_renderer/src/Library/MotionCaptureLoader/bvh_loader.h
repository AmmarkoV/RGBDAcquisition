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

/**
* @brief MAX_BVH_JOINT_HIERARCHY_SIZE is the maximum number of Joints
* @ingroup BVH
*/
#define MAX_BVH_JOINT_HIERARCHY_SIZE 256

enum CHANNEL_NAMES
{
  BVH_POSITION_NONE=0,
  BVH_ROTATION_X,
  BVH_ROTATION_Y,
  BVH_ROTATION_Z,
  BVH_POSITION_X,
  BVH_POSITION_Y,
  BVH_POSITION_Z,
  //--------------------
  BVH_VALID_CHANNEL_NAMES
};

extern const char * channelNames[];

enum CHANNEL_ROTATION_ORDER
{
  BVH_ROTATION_ORDER_NONE=0,
  BVH_ROTATION_ORDER_XYZ,
  BVH_ROTATION_ORDER_XZY,
  BVH_ROTATION_ORDER_YXZ,
  BVH_ROTATION_ORDER_YZX,
  BVH_ROTATION_ORDER_ZXY,
  BVH_ROTATION_ORDER_ZYX,
  //--------------------
  BVH_VALID_ROTATION_ORDER_NAMES
};


extern const char * rotationOrderNames[];

typedef unsigned int BVHJointID;
typedef unsigned int BVHFrameID;


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
  double staticTransformation[16];
  //--------------------
  char  channelType[BVH_VALID_CHANNEL_NAMES];
  char  channelRotationOrder;
  unsigned int loadedChannels;
  //--------------------
};



/**
* @brief The lookup table has 1:1 correspondance from joints to the given motions
* @ingroup BVH
*/
struct BVH_JointToMotion_LookupTable
{
  unsigned int jointMotionOffset;
  unsigned int channelIDMotionOffset[BVH_VALID_CHANNEL_NAMES];
};

/**
* @brief The lookup table has 1:1 correspondance from motions to the given joints
* @ingroup BVH
*/
struct BVH_MotionToJoint_LookupTable
{
  unsigned int jointID;
  unsigned int parentID;
  unsigned int channelID;
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
  struct BVH_Joint jointHierarchy[MAX_BVH_JOINT_HIERARCHY_SIZE];

  //Lookup Tables..
  struct BVH_JointToMotion_LookupTable jointToMotionLookup[MAX_BVH_JOINT_HIERARCHY_SIZE];
  struct BVH_MotionToJoint_LookupTable motionToJointLookup[MAX_BVH_JOINT_HIERARCHY_SIZE*6];

  //Motion
  unsigned int numberOfValuesPerFrame;
  unsigned int motionValuesSize;
  float * motionValues;

  //Internal Variables..
  char fileName[1024];
  unsigned int linesParsed;
};



void bvh_renameJoints(struct BVH_MotionCapture * bvhMotion);

/**
* @brief Print BVH information on stderr
* @ingroup BVH
* @param  pointer to an allocated BVH_MotionCapture struct
*/
void bvh_printBVH(struct BVH_MotionCapture * bvhMotion);

void bvh_printBVHJointToMotionLookupTable(struct BVH_MotionCapture * bvhMotion);


/**
* @brief Load a BVH file by giving a filename and filling in a BVH_MotionCapture struct
* @ingroup BVH
* @param  C-String with path to BVH File
* @param  pointer to an allocated BVH_MotionCapture struct
*/
int bvh_loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion);



int bvh_free(struct BVH_MotionCapture * bvhMotion);

/**
* @brief Resolve a C-String from a Joint name to a Joint ID
* @ingroup BVH
* @param  BVH Structure
* @param  C-String with Joint Name
* @param  Output Joint ID
*/
int bvh_getJointIDFromJointName( struct BVH_MotionCapture * bvhMotion , const char * jointName, BVHJointID * jID);



float * bvh_getJointOffset(struct BVH_MotionCapture * bvhMotion , BVHJointID jID);

float  bvh_getJointRotationXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);
float  bvh_getJointRotationYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);
float  bvh_getJointRotationZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

float  bvh_getJointPositionXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);
float  bvh_getJointPositionYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);
float  bvh_getJointPositionZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

int bhv_populatePosXYZRotXYZ(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID , float * data , unsigned int sizeOfData);


float bvh_getMotionValue(struct BVH_MotionCapture * bvhMotion , unsigned int mID);
int bhv_jointHasParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID );
int bhv_jointHasRotation(struct BVH_MotionCapture * bvhMotion , BVHJointID jID);
#endif // BVH_LOADER_H_INCLUDED