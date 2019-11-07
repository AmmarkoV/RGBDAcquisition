/** @file bvh_loader.h
 *  @brief  BVH file parser
            part of  https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer
            This is a very clean-cut parser of bvh file that will load a struct BVH_MotionCapture using the loadBVH command
            This part of the code does not perform any transformations on the loaded file, if you want to check transformations see bvh_transform.h
 *  @author Ammar Qammaz (AmmarkoV)
 */
#ifndef BVH_LOADER_H_INCLUDED
#define BVH_LOADER_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

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




enum ORIENTATION_ENFORCER_IDS
{
  BVH_ENFORCE_NO_ORIENTATION=0,
  BVH_ENFORCE_FRONT_ORIENTATION,
  BVH_ENFORCE_BACK_ORIENTATION,
  BVH_ENFORCE_LEFT_ORIENTATION,
  BVH_ENFORCE_RIGHT_ORIENTATION,
  //--------------------
  BVH_VALID_ENFORCED_ORIENTATION_NAMES
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
  char isImmuneToTorsoOcclusions;
  //--------------------
   char isAPartOfLeftFoot;
   char isAPartOfRightFoot;
   char isAPartOfLeftArm;
   char isAPartOfRightArm;
   char isAPartOfLeftHand;
   char isAPartOfRightHand;
   char isAPartOfHead;
   char isAPartOfTorso;
  //--------------------
  char isRoot;
  char hasBrace;
  char isEndSite;
  char hasEndSite;
  //--------------------
  char jointName[MAX_BVH_JOINT_NAME+1];
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
  float scaleWorld;

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




double bvh_constrainAngleCentered0(double angle,unsigned int flipOrientation);

double bvh_RemapAngleCentered0(double angle, unsigned int constrainOrientation);


int enumerateChannelOrderFromTypes(char typeA,char typeB,char typeC);

int enumerateChannelOrder(struct BVH_MotionCapture * bvhMotion , unsigned int currentJoint);

/**
* @brief Load a BVH file by giving a filename and filling in a BVH_MotionCapture struct
* @ingroup BVH
* @param  C-String with path to BVH File
* @param  pointer to an allocated BVH_MotionCapture struct
* @param  the scale of the world
*/
int bvh_loadBVH(const char * filename , struct BVH_MotionCapture * bvhMotion, float scaleWorld);


/**
* @brief Free the space allocated by a BVH_MotionCapture file
* @ingroup BVH
* @param  BVH Structure
* @return 1=Success/0=Failure
*/
int bvh_free(struct BVH_MotionCapture * bvhMotion);


int bvh_SetPositionRotation(
                            struct BVH_MotionCapture * mc,
                            float * position,
                            float * rotation
                           );


int bvh_OffsetPositionRotation(
                               struct BVH_MotionCapture * mc,
                               float * position,
                               float * rotation
                              );


int bvh_ConstrainRotations(
                           struct BVH_MotionCapture * mc,
                           unsigned int constrainOrientation
                          );



int bvh_testConstrainRotations();

int bvh_InterpolateMotion(
                           struct BVH_MotionCapture * mc
                         );


/**
* @brief Get Parent joint .
* @ingroup BVH
* @param  BVH Structure
* @param  Joint ID we want to query
* @return 1=HasParent/0=NoParent-Error
*/
int bhv_getJointParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID );



int bvh_onlyAnimateGivenJoints(struct BVH_MotionCapture * bvhMotion,unsigned int numberOfArguments,char **argv);

/**
* @brief Ask if joint has a parent.
* @ingroup BVH
* @param  BVH Structure
* @param  Joint ID we want to query
* @return 1=HasParent/0=NoParent-Error
*/
int bhv_jointHasParent(struct BVH_MotionCapture * bvhMotion , BVHJointID jID );




int bhv_jointGetEndSiteChild(struct BVH_MotionCapture * bvhMotion,BVHJointID jID,BVHJointID * jChildID);


/**
* @brief Ask if joint has rotational component.
* @ingroup BVH
* @param  BVH Structure
* @param  Joint ID we want to query
* @return 1=HasRotation/0=NoRotation-Error
*/
int bhv_jointHasRotation(struct BVH_MotionCapture * bvhMotion , BVHJointID jID);

/**
* @brief Resolve a C-String from a Joint name to a Joint ID
* @ingroup BVH
* @param  BVH Structure
* @param  C-String with Joint Name
* @param  Output Joint ID
* @return 1=Found/0=NotFound
*/
int bvh_getJointIDFromJointName( struct BVH_MotionCapture * bvhMotion , const char * jointName, BVHJointID * jID);

int bvh_getJointIDFromJointNameNocase(
                                 struct BVH_MotionCapture * bvhMotion ,
                                 const char * jointName,
                                 BVHJointID * jID
                                );

/**
* @brief Get the root joint of a BVH_MotionCapture file
* @ingroup BVH
* @param   BVH Structure
* @param   Output Root Joint ID
* @return  1=Found/0=NotFound
*/
int bvh_getRootJointID(
                       struct BVH_MotionCapture * bvhMotion ,
                       BVHJointID * jID
                      );

/**
* @brief Copy a motion frame of a BVH_MotionCapture file
* @ingroup BVH
* @param  BVH Structure
* @param  Target frame we want to populate with source
* @param  Source frame we want to copy
* @return 1=Success/0=Failure
*/
int bvh_copyMotionFrame(
                         struct BVH_MotionCapture * bvhMotion,
                         BVHFrameID tofID,
                         BVHFrameID fromfID
                        );


//float * bvh_getJointOffset(struct BVH_MotionCapture * bvhMotion , BVHJointID jID);


/**
* @brief Request a specific motion channel of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @param  Type of channel we want to access ( see enum CHANNEL_NAMES )
* @return Value of channel for specific joint at specific frame
*/
float bvh_getJointChannelAtFrame(struct BVH_MotionCapture * bvhMotion, BVHJointID jID, BVHFrameID fID, unsigned int channelTypeID);



/**
* @brief Request X Rotation of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return X Rotation for specific joint at specific frame
*/
float  bvh_getJointRotationXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request Y Rotation of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return Y Rotation for specific joint at specific frame
*/
float  bvh_getJointRotationYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request Z Rotation of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return Z Rotation for specific joint at specific frame
*/
float  bvh_getJointRotationZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request X Position of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return X Position for specific joint at specific frame
*/
float  bvh_getJointPositionXAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request Y Position of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return Y Position for specific joint at specific frame
*/
float  bvh_getJointPositionYAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);

/**
* @brief Request Z Position of a specific joint at a specific frame
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @return Z Position for specific joint at specific frame
*/
float  bvh_getJointPositionZAtFrame(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID);



/**
* @brief Request XYZ positions and XYZ rotations that will be written on a float data[6] array
* @ingroup BVH
* @param  BVH Structure
* @param  Joint we want to access ( if we just have the name we can retrieve jID using bvh_getJointIDFromJointName )
* @param  Frame we want to access
* @param  Output float array that will hold the results
* @param  Size of the output float array that should hold at least 6 floats
* @return 1=Success/0=Failure
*/
int bhv_populatePosXYZRotXYZ(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , BVHFrameID fID , float * data , unsigned int sizeOfData);

int bhv_populatePosXYZRotXYZFromMotionBuffer(struct BVH_MotionCapture * bvhMotion , BVHJointID jID , float * motionBuffer, float * data, unsigned int sizeOfData);

/**
* @brief Direct access to the motion data, without Joint hierarchy,Frame separation etc, should not be used unless you really know what you are doing..
* @ingroup BVH
* @param  BVH Structure
* @param  Motion element we want to read
* @return Motion Value
*/
float bvh_getMotionValue(struct BVH_MotionCapture * bvhMotion , unsigned int mID);






/**
* @brief Print BVH information on stderr
* @ingroup BVH
* @param  pointer to an allocated BVH_MotionCapture struct
*/
void bvh_printBVH(struct BVH_MotionCapture * bvhMotion);

void bvh_printBVHJointToMotionLookupTable(struct BVH_MotionCapture * bvhMotion);


#ifdef __cplusplus
}
#endif




#endif // BVH_LOADER_H_INCLUDED
