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


void printSkeletonState(nite::UserTracker & pUserTracker , const nite::UserData & user)
{

	float bbox[] =
	{
		user.getBoundingBox().max.x, user.getBoundingBox().max.y, 0,
		user.getBoundingBox().max.x, user.getBoundingBox().min.y, 0,
		user.getBoundingBox().min.x, user.getBoundingBox().min.y, 0,
		user.getBoundingBox().min.x, user.getBoundingBox().max.y, 0,
	};

	fprintf(stderr,"Bounding Box %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n",bbox[0],bbox[1],bbox[2],bbox[3],bbox[4],bbox[5],bbox[6],bbox[7]);
    fprintf(stderr,"Center mass %0.2f %0.2f %0.2f \n",user.getCenterOfMass().x , user.getCenterOfMass().y, user.getCenterOfMass().z);

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


     fprintf(stderr,"Head %f %f %f \n",jointHead.getPosition().x ,jointHead.getPosition().y, jointHead.getPosition().z);


     //This only works for 2 poses and needs initialization fprintf(stderr,"Pose %u ( held %u , enter %u , exit %u)\n",user.getPose().getType() , user.getPose().isHeld() , user.getPose().isEntered() , user.getPose().isExited() );
}






void updateUserState(nite::UserTracker & pUserTracker ,const nite::UserData& user, unsigned long long ts)
{
	if (user.isNew())
		USER_MESSAGE("New")
	else if (user.isVisible() && !g_visibleUsers[user.getId()])
		USER_MESSAGE("Visible")
	else if (!user.isVisible() && g_visibleUsers[user.getId()])
		USER_MESSAGE("Out of Scene")
	else if (user.isLost())
		USER_MESSAGE("Lost")

	g_visibleUsers[user.getId()] = user.isVisible();


	if(g_skeletonStates[user.getId()] != user.getSkeleton().getState())
	{
		switch(g_skeletonStates[user.getId()] = user.getSkeleton().getState())
		{
		case nite::SKELETON_NONE:
			USER_MESSAGE("Stopped tracking.")
			break;
		case nite::SKELETON_CALIBRATING:
			USER_MESSAGE("Calibrating...")
			break;
		case nite::SKELETON_TRACKED:
			USER_MESSAGE("Tracking!")
			break;
		case nite::SKELETON_CALIBRATION_ERROR_NOT_IN_POSE:
		case nite::SKELETON_CALIBRATION_ERROR_HANDS:
		case nite::SKELETON_CALIBRATION_ERROR_LEGS:
		case nite::SKELETON_CALIBRATION_ERROR_HEAD:
		case nite::SKELETON_CALIBRATION_ERROR_TORSO:
			USER_MESSAGE("Calibration Failed... :-|")
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



int loopNite2()
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
			updateUserState(userTracker,user,userTrackerFrame.getTimestamp());
			if (user.isNew())
			{
				userTracker.startSkeletonTracking(user.getId());
			}
			else if (user.getSkeleton().getState() == nite::SKELETON_TRACKED)
			{
				const nite::SkeletonJoint& head = user.getSkeleton().getJoint(nite::JOINT_HEAD);
				if (head.getPositionConfidence() > .5)
				 {
				     printf("%d. (%5.2f, %5.2f, %5.2f)\n", user.getId(), head.getPosition().x, head.getPosition().y, head.getPosition().z);
                     printSkeletonState(userTracker,user);
				 }
			}
		}
  return 1;
}
