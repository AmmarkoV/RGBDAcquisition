/*******************************************************************************
*                                                                              *
*   PrimeSense NiTE 2.0 - Simple Skeleton Sample                               *
*   Copyright (C) 2012 PrimeSense Ltd.                                         *
*                                                                              *
*******************************************************************************/

#include "NiTE.h"

#include <NiteSampleUtilities.h>

#define MAX_USERS 10
bool g_visibleUsers[MAX_USERS] = {false};
nite::SkeletonState g_skeletonStates[MAX_USERS] = {nite::SKELETON_NONE};

#define USER_MESSAGE(msg) \
	{printf("[%08llu] User #%d:\t%s\n",ts, user.getId(),msg);}

	nite::UserTracker userTracker;
	nite::Status niteRc;
	nite::UserTrackerFrameRef userTrackerFrame;



void * skelCallbackAddr = 0;


enum humanSkeletonJoints
{
   HUMAN_SKELETON_HEAD = 0,
   HUMAN_SKELETON_LEFT_SHOULDER,
   HUMAN_SKELETON_RIGHT_SHOULDER,
   HUMAN_SKELETON_LEFT_ELBOW,
   HUMAN_SKELETON_RIGHT_ELBOW,
   HUMAN_SKELETON_LEFT_HAND,
   HUMAN_SKELETON_RIGHT_HAND,
   HUMAN_SKELETON_TORSO,
   HUMAN_SKELETON_LEFT_HIP,
   HUMAN_SKELETON_RIGHT_HIP,
   HUMAN_SKELETON_LEFT_KNEE,
   HUMAN_SKELETON_RIGHT_KNEE,
   HUMAN_SKELETON_LEFT_FOOT,
   HUMAN_SKELETON_RIGHT_FOOT,
   //---------------------
   HUMAN_SKELETON_PARTS
};

struct point3D
{
    float x,y,z;
};

struct skeletonHuman
{
  unsigned int userID;

  unsigned char isNew,isVisible,isOutOfScene,isLost;
  unsigned char statusCalibrating,statusStoppedTracking, statusTracking,statusFailed;

  struct point3D bbox[4];
  struct point3D centerOfMass;
  struct point3D joint[HUMAN_SKELETON_PARTS];
  float jointAccuracy[HUMAN_SKELETON_PARTS];
};


void printSkeletonState(unsigned int frameNumber , nite::UserTracker & pUserTracker , const nite::UserData & user)
{
    struct skeletonHuman humanSkeleton={0};

    humanSkeleton.bbox[0].x = user.getBoundingBox().max.x;       humanSkeleton.bbox[0].y = user.getBoundingBox().max.y;   humanSkeleton.bbox[0].z = 0;
    humanSkeleton.bbox[1].x = user.getBoundingBox().max.x;       humanSkeleton.bbox[1].y = user.getBoundingBox().min.y;   humanSkeleton.bbox[1].z = 0;
    humanSkeleton.bbox[2].x = user.getBoundingBox().min.x;       humanSkeleton.bbox[2].y = user.getBoundingBox().min.y;   humanSkeleton.bbox[2].z = 0;
    humanSkeleton.bbox[3].x = user.getBoundingBox().min.x;       humanSkeleton.bbox[3].y = user.getBoundingBox().max.y;   humanSkeleton.bbox[3].z = 0;

    humanSkeleton.userID = user.getId();

    humanSkeleton.centerOfMass.x = user.getCenterOfMass().x;
    humanSkeleton.centerOfMass.y = user.getCenterOfMass().y;
    humanSkeleton.centerOfMass.z = user.getCenterOfMass().z;

     nite::SkeletonJoint jointHead           =   user.getSkeleton().getJoint(nite::JOINT_HEAD);
     nite::SkeletonJoint jointLeftShoulder   =   user.getSkeleton().getJoint(nite::JOINT_LEFT_SHOULDER);
     nite::SkeletonJoint jointRightShoulder  =   user.getSkeleton().getJoint(nite::JOINT_RIGHT_SHOULDER);
     nite::SkeletonJoint jointLeftElbow      =   user.getSkeleton().getJoint(nite::JOINT_LEFT_ELBOW);
     nite::SkeletonJoint jointRightElbow     =   user.getSkeleton().getJoint(nite::JOINT_RIGHT_ELBOW);
     nite::SkeletonJoint jointLeftHand      =   user.getSkeleton().getJoint(nite::JOINT_LEFT_HAND);
     nite::SkeletonJoint jointRightHand      =   user.getSkeleton().getJoint(nite::JOINT_RIGHT_HAND);
     nite::SkeletonJoint jointTorso     =   user.getSkeleton().getJoint(nite::JOINT_TORSO);
     nite::SkeletonJoint jointLeftHip     =   user.getSkeleton().getJoint(nite::JOINT_LEFT_HIP);
     nite::SkeletonJoint jointRightHip     =   user.getSkeleton().getJoint(nite::JOINT_RIGHT_HIP);
     nite::SkeletonJoint jointLeftKnee     =   user.getSkeleton().getJoint(nite::JOINT_LEFT_KNEE);
     nite::SkeletonJoint jointRightKnee     =   user.getSkeleton().getJoint(nite::JOINT_RIGHT_KNEE);
     nite::SkeletonJoint jointLeftFoot     =   user.getSkeleton().getJoint(nite::JOINT_LEFT_FOOT);
     nite::SkeletonJoint jointRightFoot     =   user.getSkeleton().getJoint(nite::JOINT_RIGHT_FOOT);

     humanSkeleton.joint[HUMAN_SKELETON_HEAD].x = jointHead.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_HEAD].y = jointHead.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_HEAD].z = jointHead.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_HEAD] = jointHead.getPositionConfidence();


     humanSkeleton.joint[HUMAN_SKELETON_LEFT_SHOULDER].x = jointLeftShoulder.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_SHOULDER].y = jointLeftShoulder.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_SHOULDER].z = jointLeftShoulder.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_SHOULDER] = jointLeftShoulder.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_SHOULDER].x = jointRightShoulder.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_SHOULDER].y = jointRightShoulder.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_SHOULDER].z = jointRightShoulder.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_SHOULDER] = jointRightShoulder.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_ELBOW].x = jointLeftElbow.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_ELBOW].y = jointLeftElbow.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_ELBOW].z = jointLeftElbow.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_ELBOW] = jointLeftElbow.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_ELBOW].x = jointRightElbow.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_ELBOW].y = jointRightElbow.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_ELBOW].z = jointRightElbow.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_ELBOW] = jointRightElbow.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HAND].x = jointLeftHand.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HAND].y = jointLeftHand.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HAND].z = jointLeftHand.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_HAND] = jointLeftHand.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HAND].x = jointRightHand.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HAND].y = jointRightHand.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HAND].z = jointRightHand.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_HAND] = jointRightHand.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_TORSO].x = jointTorso.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_TORSO].y = jointTorso.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_TORSO].z = jointTorso.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_TORSO] = jointTorso.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HIP].x = jointLeftHip.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HIP].y = jointLeftHip.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HIP].z = jointLeftHip.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_HIP] = jointLeftHip.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HIP].x = jointRightHip.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HIP].y = jointRightHip.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HIP].z = jointRightHip.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_HIP] = jointRightHip.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_KNEE].x = jointLeftKnee.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_KNEE].y = jointLeftKnee.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_KNEE].z = jointLeftKnee.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_KNEE] = jointLeftKnee.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_KNEE].x = jointRightKnee.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_KNEE].y = jointRightKnee.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_KNEE].z = jointRightKnee.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_KNEE] = jointRightKnee.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_FOOT].x = jointLeftFoot.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_FOOT].y = jointLeftFoot.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_FOOT].z = jointLeftFoot.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_FOOT] = jointLeftFoot.getPositionConfidence();

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_FOOT].x = jointRightFoot.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_FOOT].y = jointRightFoot.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_FOOT].z = jointRightFoot.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_FOOT] = jointRightFoot.getPositionConfidence();



  unsigned char statusCalibrating,statusTracking,statusFailed;

    long long unsigned int ts = frameNumber;
	if (user.isNew())  { humanSkeleton.isNew=1; } else
    if ((user.isVisible()) && (!g_visibleUsers[user.getId()])) { humanSkeleton.isVisible=1; }  else
    if ((!user.isVisible()) && (g_visibleUsers[user.getId()])) { humanSkeleton.isOutOfScene=1; } else
    if (user.isLost())  { humanSkeleton.isLost=1; }

	g_visibleUsers[user.getId()] = user.isVisible();
	if(g_skeletonStates[user.getId()] != user.getSkeleton().getState())
	{
		switch(g_skeletonStates[user.getId()] = user.getSkeleton().getState())
		{
		 case nite::SKELETON_NONE:         humanSkeleton.statusStoppedTracking = 1; break; // USER_MESSAGE("Stopped tracking.")
		 case nite::SKELETON_CALIBRATING:  humanSkeleton.statusCalibrating=1;       break; // USER_MESSAGE("Calibrating...")
	 	 case nite::SKELETON_TRACKED:      humanSkeleton.statusTracking=1;          break; // USER_MESSAGE("Tracking!")
		 case nite::SKELETON_CALIBRATION_ERROR_NOT_IN_POSE:
		 case nite::SKELETON_CALIBRATION_ERROR_HANDS:
		 case nite::SKELETON_CALIBRATION_ERROR_LEGS:
		 case nite::SKELETON_CALIBRATION_ERROR_HEAD:
		 case nite::SKELETON_CALIBRATION_ERROR_TORSO:
			humanSkeleton.statusFailed=1; //USER_MESSAGE("Calibration Failed... :-|")
		 break;
		}
	}

 }




int startNite2()
{
    printf("Starting Nite2\n");
	nite::NiTE::initialize();

	niteRc = userTracker.create();
	if (niteRc != nite::STATUS_OK)
	{
		printf("Couldn't create user tracker\n");
		return 3;
	}
	printf("\nStart moving around to get detected...\n(PSI pose may be required for skeleton calibration, depending on the configuration)\n");

 return 1;
}


int stopNite2()
{
	nite::NiTE::shutdown();
  return 1;
}



int loopNite2(unsigned int frameNumber)
{
		niteRc = userTracker.readFrame(&userTrackerFrame);
		if (niteRc != nite::STATUS_OK)
		{
			printf("Get next frame failed\n");
			return 0;
		}

		const nite::Array<nite::UserData>& users = userTrackerFrame.getUsers();
		for (int i = 0; i < users.getSize(); ++i)
		{
			const nite::UserData& user = users[i];
			//updateUserState(frameNumber,userTracker,user,userTrackerFrame.getTimestamp());
			if (user.isNew())
			{
				userTracker.startSkeletonTracking(user.getId());
			}
			else if (user.getSkeleton().getState() == nite::SKELETON_TRACKED)
			{
				const nite::SkeletonJoint& head = user.getSkeleton().getJoint(nite::JOINT_HEAD);
				if (head.getPositionConfidence() > .5)
				 {
				     //printf("%d. (%5.2f, %5.2f, %5.2f)\n", user.getId(), head.getPosition().x, head.getPosition().y, head.getPosition().z);
                     printSkeletonState(frameNumber,userTracker,user);
				 }
			}
		}
  return 1;
}

int registerSkeletonDetectedEvent(void * callback)
{
  skelCallbackAddr  = callback;
  return 1;
}


