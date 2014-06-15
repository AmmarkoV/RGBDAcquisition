#ifndef NITE2_H_INCLUDED
#define NITE2_H_INCLUDED


#include <OpenNI.h>
#include <PS1080.h>

using namespace openni;


static const char * jointNames[] =
{"head",
 "neck",
 "torso",
 "left_shoulder",
 "right_shoulder",
 "left_elbow",
 "right_elbow",
 "left_hand",
 "right_hand",
 "left_hip",
 "right_hip",
 "left_knee",
 "right_knee",
 "left_foot",
 "right_foot"
};


static const char * const humanSkeletonJointNames[] =
    {
       "HUMAN_SKELETON_HEAD",
       "HUMAN_SKELETON_NECK",
       "HUMAN_SKELETON_TORSO",
       "HUMAN_SKELETON_RIGHT_SHOULDER",
       "HUMAN_SKELETON_LEFT_SHOULDER",
       "HUMAN_SKELETON_RIGHT_ELBOW",
       "HUMAN_SKELETON_LEFT_ELBOW",
       "HUMAN_SKELETON_RIGHT_HAND",
       "HUMAN_SKELETON_LEFT_HAND",
       "HUMAN_SKELETON_RIGHT_HIP",
       "HUMAN_SKELETON_LEFT_HIP",
       "HUMAN_SKELETON_RIGHT_KNEE",
       "HUMAN_SKELETON_LEFT_KNEE",
       "HUMAN_SKELETON_RIGHT_FOOT",
       "HUMAN_SKELETON_LEFT_FOOT"
    };

enum humanSkeletonJoints
{
   HUMAN_SKELETON_HEAD = 0,
   HUMAN_SKELETON_NECK,
   HUMAN_SKELETON_TORSO,
   HUMAN_SKELETON_RIGHT_SHOULDER,
   HUMAN_SKELETON_LEFT_SHOULDER,
   HUMAN_SKELETON_RIGHT_ELBOW,
   HUMAN_SKELETON_LEFT_ELBOW,
   HUMAN_SKELETON_RIGHT_HAND,
   HUMAN_SKELETON_LEFT_HAND,
   HUMAN_SKELETON_RIGHT_HIP,
   HUMAN_SKELETON_LEFT_HIP,
   HUMAN_SKELETON_RIGHT_KNEE,
   HUMAN_SKELETON_LEFT_KNEE,
   HUMAN_SKELETON_RIGHT_FOOT,
   HUMAN_SKELETON_LEFT_FOOT,
   //---------------------
   HUMAN_SKELETON_PARTS
};

static const char * const humanSkeletonMirroredJointNames[] =
    {
      "HUMAN_SKELETON_MIRRORED_HEAD",
      "HUMAN_SKELETON_MIRRORED_NECK",
      "HUMAN_SKELETON_MIRRORED_TORSO",
      "HUMAN_SKELETON_MIRRORED_LEFT_SHOULDER",
      "HUMAN_SKELETON_MIRRORED_RIGHT_SHOULDER",
      "HUMAN_SKELETON_MIRRORED_LEFT_ELBOW",
      "HUMAN_SKELETON_MIRRORED_RIGHT_ELBOW",
      "HUMAN_SKELETON_MIRRORED_LEFT_HAND",
      "HUMAN_SKELETON_MIRRORED_RIGHT_HAND",
      "HUMAN_SKELETON_MIRRORED_LEFT_HIP",
      "HUMAN_SKELETON_MIRRORED_RIGHT_HIP",
      "HUMAN_SKELETON_MIRRORED_LEFT_KNEE",
      "HUMAN_SKELETON_MIRRORED_RIGHT_KNEE",
      "HUMAN_SKELETON_MIRRORED_LEFT_FOOT",
      "HUMAN_SKELETON_MIRRORED_RIGHT_FOOT",
    };

enum humanMirroredSkeletonJoints
{
   HUMAN_SKELETON_MIRRORED_HEAD = 0,
   HUMAN_SKELETON_MIRRORED_NECK,
   HUMAN_SKELETON_MIRRORED_TORSO,
   HUMAN_SKELETON_MIRRORED_LEFT_SHOULDER,
   HUMAN_SKELETON_MIRRORED_RIGHT_SHOULDER,
   HUMAN_SKELETON_MIRRORED_LEFT_ELBOW,
   HUMAN_SKELETON_MIRRORED_RIGHT_ELBOW,
   HUMAN_SKELETON_MIRRORED_LEFT_HAND,
   HUMAN_SKELETON_MIRRORED_RIGHT_HAND,
   HUMAN_SKELETON_MIRRORED_LEFT_HIP,
   HUMAN_SKELETON_MIRRORED_RIGHT_HIP,
   HUMAN_SKELETON_MIRRORED_LEFT_KNEE,
   HUMAN_SKELETON_MIRRORED_RIGHT_KNEE,
   HUMAN_SKELETON_MIRRORED_LEFT_FOOT,
   HUMAN_SKELETON_MIRRORED_RIGHT_FOOT,
   //---------------------
   HUMAN_SKELETON_MIRRORED_PARTS
};

extern const char * jointNames[];

struct point3D
{
    float x,y,z;
};

struct point2D
{
    float x,y;
};


struct skeletonHuman
{
  unsigned int observationNumber , observationTotal;
  unsigned int userID;

  unsigned char isNew,isVisible,isOutOfScene,isLost;
  unsigned char statusCalibrating,statusStoppedTracking, statusTracking,statusFailed;

  struct point3D bbox[8];
  struct point3D bboxDimensions;
  struct point3D centerOfMass;
  struct point3D joint[HUMAN_SKELETON_PARTS];
  struct point2D joint2D[HUMAN_SKELETON_PARTS];
  float jointAccuracy[HUMAN_SKELETON_PARTS];
};


struct skeletonPointing
{
  struct point3D pointStart;
  struct point3D pointEnd;
  struct point3D pointingVector;
  unsigned char isLeftHand;
  unsigned char isRightHand;
};


int registerSkeletonPointingDetectedEvent(int devID,void * callback);
int registerSkeletonDetectedEvent(int devID,void * callback);

int startNite2(int maxVirtualSkeletonTrackers);
int createNite2Device(int devID,openni::Device * device);
int destroyNite2Device(int devID);

int stopNite2();
int loopNite2(int devID,unsigned int frameNumber);

unsigned short  * getNite2DepthFrame(int devID);
int getNite2DepthHeight(int devID);
int getNite2DepthWidth(int devID);

#endif // NITE2_H_INCLUDED
