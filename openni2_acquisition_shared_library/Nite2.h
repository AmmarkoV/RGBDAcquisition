#ifndef NITE2_H_INCLUDED
#define NITE2_H_INCLUDED


#include <OpenNI.h>
#include <PS1080.h>

using namespace openni;




enum humanSkeletonJoints
{
   HUMAN_SKELETON_HEAD = 0,
   HUMAN_SKELETON_TORSO,
   HUMAN_SKELETON_LEFT_SHOULDER,
   HUMAN_SKELETON_RIGHT_SHOULDER,
   HUMAN_SKELETON_LEFT_ELBOW,
   HUMAN_SKELETON_RIGHT_ELBOW,
   HUMAN_SKELETON_LEFT_HAND,
   HUMAN_SKELETON_RIGHT_HAND,
   HUMAN_SKELETON_LEFT_HIP,
   HUMAN_SKELETON_RIGHT_HIP,
   HUMAN_SKELETON_LEFT_KNEE,
   HUMAN_SKELETON_RIGHT_KNEE,
   HUMAN_SKELETON_LEFT_FOOT,
   HUMAN_SKELETON_RIGHT_FOOT,
   //---------------------
   HUMAN_SKELETON_PARTS
};


extern const char * jointNames[];




struct point3D
{
    float x,y,z;
};

struct skeletonHuman
{
  unsigned int observationNumber , observationTotal;
  unsigned int userID;

  unsigned char isNew,isVisible,isOutOfScene,isLost;
  unsigned char statusCalibrating,statusStoppedTracking, statusTracking,statusFailed;

  struct point3D bbox[4];
  struct point3D centerOfMass;
  struct point3D joint[HUMAN_SKELETON_PARTS];
  float jointAccuracy[HUMAN_SKELETON_PARTS];
};


int registerSkeletonDetectedEvent(void * callback);

int startNite2Void();
int startNite2(Device * device);

int stopNite2();
int loopNite2(unsigned int frameNumber);


int registerSkeletonDetectedEvent(void * callback);

#endif // NITE2_H_INCLUDED
