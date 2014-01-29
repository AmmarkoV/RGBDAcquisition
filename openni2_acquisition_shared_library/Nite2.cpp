/*******************************************************************************
*                                                                              *
*   PrimeSense NiTE 2.0 - Simple Skeleton Sample                               *
*   Copyright (C) 2012 PrimeSense Ltd.                                         *
*                                                                              *
*******************************************************************************/

#include "NiTE.h"

#include "Nite2.h"
#include <math.h>

#define MAX_USERS 10

#define NORMAL "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */

const char * jointNames[] =
{"head",
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

bool g_visibleUsers[MAX_USERS] = {false};
nite::SkeletonState g_skeletonStates[MAX_USERS] = {nite::SKELETON_NONE};

#define USER_MESSAGE(msg) \
	{printf("[%08llu] User #%d:\t%s\n",ts, user.getId(),msg);}

	nite::UserTracker userTracker;
	nite::Status niteRc;
	nite::UserTrackerFrameRef userTrackerFrame;



void * skelCallbackAddr = 0;
void * skelCallbackPointingAddr = 0;



int registerSkeletonPointingDetectedEvent(void * callback)
{
  skelCallbackPointingAddr=callback;
  return 1;
}

int registerSkeletonDetectedEvent(void * callback)
{
  skelCallbackAddr = callback;
  return 1;
}




void newSkeletonPointingDetected(unsigned int frameNumber ,struct skeletonPointing * skeletonPointingFound)
{

  fprintf(stderr,YELLOW "Skeleton Pointing Detected : ");
  if (skeletonPointingFound->isLeftHand) {  fprintf(stderr,"LEFT "); } else
  if (skeletonPointingFound->isRightHand) {  fprintf(stderr,"RIGHT "); }
  fprintf(stderr," From %0.2f,%0.2f,%0.2f\n",skeletonPointingFound->pointStart.x,skeletonPointingFound->pointStart.y,skeletonPointingFound->pointStart.z);
  fprintf(stderr," To   %0.2f,%0.2f,%0.2f\n",skeletonPointingFound->pointEnd.x,skeletonPointingFound->pointEnd.y,skeletonPointingFound->pointEnd.z);
  fprintf(stderr," Vector %0.2f,%0.2f,%0.2f\n",skeletonPointingFound->pointingVector.x,skeletonPointingFound->pointingVector.y,skeletonPointingFound->pointingVector.z);
  fprintf(stderr,"\n " NORMAL);



  if (skelCallbackPointingAddr!=0)
  {
    void ( *DoCallback) (unsigned int ,struct skeletonPointing *)=0 ;
    DoCallback = (void(*) (unsigned int ,struct skeletonPointing *) ) skelCallbackPointingAddr;
    DoCallback(frameNumber ,skeletonPointingFound);
  }

}


void newSkeletonDetected(unsigned int frameNumber ,struct skeletonHuman * skeletonFound)
{
    fprintf(stderr, GREEN " " );
    fprintf(stderr,"Skeleton #%u found at frame %u \n",skeletonFound->userID, frameNumber);
    unsigned int i=0;

    fprintf(stderr,"BBox : ( ");
    for (int i=0; i<4; i++)
    {
      fprintf(stderr,"%0.1f %0.1f , " ,skeletonFound->bbox[i].x , skeletonFound->bbox[i].y  );
    }
    fprintf(stderr,"\n");

    fprintf(stderr,"Center of Mass %0.2f %0.2f %0.2f \n",skeletonFound->centerOfMass.x,skeletonFound->centerOfMass.y,skeletonFound->centerOfMass.z);


    fprintf(stderr,"Head %0.2f %0.2f %0.2f \n",skeletonFound->joint[HUMAN_SKELETON_HEAD].x,skeletonFound->joint[HUMAN_SKELETON_HEAD].y,skeletonFound->joint[HUMAN_SKELETON_HEAD].z);

    fprintf(stderr,  " \n" NORMAL );


  if (skelCallbackAddr!=0)
  {
    void ( *DoCallback) (unsigned int ,struct skeletonHuman *)=0 ;
    DoCallback = (void(*) (unsigned int ,struct skeletonHuman *) ) skelCallbackAddr;
    DoCallback(frameNumber ,skeletonFound);
  }

}



float simpPow(float base,unsigned int exp)
{
    if (exp==0) return 1;
    float retres=base;
    unsigned int i=0;
    for (i=0; i<exp-1; i++)
    {
        retres*=base;
    }
    return retres;
}



int considerSkeletonPointing(unsigned int frameNumber,struct skeletonHuman * skeletonFound)
{
  struct skeletonPointing skelPF={0};

  float distanceLeft = sqrt(
                             simpPow(skeletonFound->joint[HUMAN_SKELETON_TORSO].x - skeletonFound->joint[HUMAN_SKELETON_LEFT_HAND].x ,2)  +
                             simpPow(skeletonFound->joint[HUMAN_SKELETON_TORSO].y - skeletonFound->joint[HUMAN_SKELETON_LEFT_HAND].y ,2)  +
                             simpPow(skeletonFound->joint[HUMAN_SKELETON_TORSO].z - skeletonFound->joint[HUMAN_SKELETON_LEFT_HAND].z ,2)
                           );
  float distanceRight = sqrt(
                             simpPow(skeletonFound->joint[HUMAN_SKELETON_TORSO].x - skeletonFound->joint[HUMAN_SKELETON_RIGHT_HAND].x ,2)  +
                             simpPow(skeletonFound->joint[HUMAN_SKELETON_TORSO].y - skeletonFound->joint[HUMAN_SKELETON_RIGHT_HAND].y ,2)  +
                             simpPow(skeletonFound->joint[HUMAN_SKELETON_TORSO].z - skeletonFound->joint[HUMAN_SKELETON_RIGHT_HAND].z ,2)
                             );


  if ( (distanceLeft<300) && (distanceRight<300) ) { fprintf(stderr,"Cutting off pointing "); return 0; }


  int doHand=1; //1 = left , 2 =right
  if (distanceLeft<distanceRight) { doHand=2; }

  if (doHand==1)
  {
   skelPF.pointStart.x = skeletonFound->joint[HUMAN_SKELETON_LEFT_ELBOW].x;
   skelPF.pointStart.y = skeletonFound->joint[HUMAN_SKELETON_LEFT_ELBOW].y;
   skelPF.pointStart.z = skeletonFound->joint[HUMAN_SKELETON_LEFT_ELBOW].z;
   skelPF.pointEnd.x = skeletonFound->joint[HUMAN_SKELETON_LEFT_HAND].x;
   skelPF.pointEnd.y = skeletonFound->joint[HUMAN_SKELETON_LEFT_HAND].y;
   skelPF.pointEnd.z = skeletonFound->joint[HUMAN_SKELETON_LEFT_HAND].z;
   skelPF.pointingVector.x = skelPF.pointEnd.x - skelPF.pointStart.x;
   skelPF.pointingVector.y = skelPF.pointEnd.y - skelPF.pointStart.y;
   skelPF.pointingVector.z = skelPF.pointEnd.z - skelPF.pointStart.z;
   skelPF.isLeftHand=1;
   skelPF.isRightHand=0;
   newSkeletonPointingDetected(frameNumber,&skelPF);
   return 1;
  } else
  if (doHand==2)
  {
   skelPF.pointStart.x = skeletonFound->joint[HUMAN_SKELETON_RIGHT_ELBOW].x;
   skelPF.pointStart.y = skeletonFound->joint[HUMAN_SKELETON_RIGHT_ELBOW].y;
   skelPF.pointStart.z = skeletonFound->joint[HUMAN_SKELETON_RIGHT_ELBOW].z;
   skelPF.pointEnd.x = skeletonFound->joint[HUMAN_SKELETON_RIGHT_HAND].x;
   skelPF.pointEnd.y = skeletonFound->joint[HUMAN_SKELETON_RIGHT_HAND].y;
   skelPF.pointEnd.z = skeletonFound->joint[HUMAN_SKELETON_RIGHT_HAND].z;
   skelPF.pointingVector.x = skelPF.pointEnd.x - skelPF.pointStart.x;
   skelPF.pointingVector.y = skelPF.pointEnd.y - skelPF.pointStart.y;
   skelPF.pointingVector.z = skelPF.pointEnd.z - skelPF.pointStart.z;
   skelPF.isLeftHand=0;
   skelPF.isRightHand=1;
   newSkeletonPointingDetected(frameNumber,&skelPF);
   return 1;
  }

 return 0;
}



void prepareSkeletonState(unsigned int frameNumber , nite::UserTracker & pUserTracker , const nite::UserData & user  , unsigned int observation , unsigned int totalObservations)
{
    struct skeletonHuman humanSkeleton={0};

    humanSkeleton.observationNumber = observation;
    humanSkeleton.observationTotal = totalObservations;

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

   //This is an event that gets fed with our newly encapsulated data
   //it should also fire up any additional events registered by clients
   newSkeletonDetected(frameNumber,&humanSkeleton);


   if (considerSkeletonPointing(frameNumber,&humanSkeleton))
   {
      fprintf(stderr,"New pointing gesture found\n");
   }

 }


int startNite2Void()
{
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

int startNite2(openni::Device * device)
{
    printf("Starting Nite2\n");
	nite::NiTE::initialize();

	niteRc = userTracker.create(device);
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
		    //We will use user from now on as the current user
			const nite::UserData& user = users[i];

            //If our user is new we should start to track his skeleton
			if (user.isNew())
			{
				userTracker.startSkeletonTracking(user.getId());
			}
			else
            //If we have a skeleton tracked , populate our internal structures and call callbacks
            if (user.getSkeleton().getState() == nite::SKELETON_TRACKED)
			{
		      prepareSkeletonState(frameNumber,userTracker,user  , i , users.getSize() );
			}
		}
  return 1;
}


