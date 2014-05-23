/*******************************************************************************
*                                                                              *
*   PrimeSense NiTE 2.0 - Simple Skeleton Sample                               *
*   Copyright (C) 2012 PrimeSense Ltd.                                         *
*                                                                              *
*******************************************************************************/

#include "NiTE.h"

#include "Nite2.h"
#include <math.h>

#define MAXIMUM_DISTANCE_FOR_POINTING 400
#define MAX_USERS 10
#define CALCULATE_BOUNDING_BOX 1

#define NORMAL "\033[0m"
#define BLACK "\033[30m" /* Black */
#define RED "\033[31m" /* Red */
#define GREEN "\033[32m" /* Green */
#define YELLOW "\033[33m" /* Yellow */
#define BLUE "\033[34m" /* Blue */
#define MAGENTA "\033[35m" /* Magenta */
#define CYAN "\033[36m" /* Cyan */
#define WHITE "\033[37m" /* White */

  #define USER_MESSAGE(msg) \
	  {printf("[%08llu] User #%d:\t%s\n",ts, user.getId(),msg);}


struct NiteVirtualDevice
{
  int failed;
  nite::UserTracker * userTracker;
  nite::UserTrackerFrameRef * userTrackerFrame;

  bool g_visibleUsers[MAX_USERS];
  nite::SkeletonState g_skeletonStates[MAX_USERS];

  void * skelCallbackAddr;
  void * skelCallbackPointingAddr;
};

//Skeleton Tracker Context shorthand
struct NiteVirtualDevice * stc=0;



void newSkeletonPointingDetected(int devID, unsigned int frameNumber ,struct skeletonPointing * skeletonPointingFound)
{
  fprintf(stderr,YELLOW "Skeleton Pointing Detected : ");
  if (skeletonPointingFound->isLeftHand) {  fprintf(stderr,"LEFT "); } else
  if (skeletonPointingFound->isRightHand) {  fprintf(stderr,"RIGHT "); }
  fprintf(stderr," From %0.2f,%0.2f,%0.2f\n",skeletonPointingFound->pointStart.x,skeletonPointingFound->pointStart.y,skeletonPointingFound->pointStart.z);
  fprintf(stderr," To   %0.2f,%0.2f,%0.2f\n",skeletonPointingFound->pointEnd.x,skeletonPointingFound->pointEnd.y,skeletonPointingFound->pointEnd.z);
  fprintf(stderr," Vector %0.2f,%0.2f,%0.2f\n",skeletonPointingFound->pointingVector.x,skeletonPointingFound->pointingVector.y,skeletonPointingFound->pointingVector.z);
  fprintf(stderr,"\n " NORMAL);



  if (stc[devID].skelCallbackPointingAddr!=0)
  {
    void ( *DoCallback) (unsigned int ,struct skeletonPointing *)=0 ;
    DoCallback = (void(*) (unsigned int ,struct skeletonPointing *) ) stc[devID].skelCallbackPointingAddr;
    DoCallback(frameNumber ,skeletonPointingFound);
  }

}


void newSkeletonDetected(int devID,unsigned int frameNumber ,struct skeletonHuman * skeletonFound)
{
    fprintf(stderr, GREEN " " );
    fprintf(stderr,"Skeleton #%u found at frame %u \n",skeletonFound->userID, frameNumber);

    fprintf(stderr,"BBox : ( ");
    unsigned int i=0;
    for (i=0; i<4; i++)
    {
      fprintf(stderr,"%0.1f %0.1f " ,skeletonFound->bbox[i].x , skeletonFound->bbox[i].y  );
      if (i<3) { fprintf(stderr,","); } else { fprintf(stderr,")"); }
    }
    fprintf(stderr,"\n");
    fprintf(stderr,"Center of Mass %0.2f %0.2f %0.2f \n",skeletonFound->centerOfMass.x,skeletonFound->centerOfMass.y,skeletonFound->centerOfMass.z);
    fprintf(stderr,"Head %0.2f %0.2f %0.2f \n",skeletonFound->joint[HUMAN_SKELETON_HEAD].x,skeletonFound->joint[HUMAN_SKELETON_HEAD].y,skeletonFound->joint[HUMAN_SKELETON_HEAD].z);
    fprintf(stderr,  " \n" NORMAL );


    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
      printf("%0.2f %0.2f ", skeletonFound->joint2D[i].x , skeletonFound->joint2D[i].y );
      //printf("JOINT2D(%s,%0.2f,%0.2f)\n" , humanSkeletonJointNames[i] , skeletonFound->joint2D[i].x , skeletonFound->joint2D[i].y );
    }
   printf("\n\n");

  if (stc[devID].skelCallbackAddr!=0)
  {
    void ( *DoCallback) (unsigned int ,struct skeletonHuman *)=0 ;
    DoCallback = (void(*) (unsigned int ,struct skeletonHuman *) ) stc[devID].skelCallbackAddr;
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



int considerSkeletonPointing(int devID ,unsigned int frameNumber,struct skeletonHuman * skeletonFound)
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


  if ( (distanceLeft<MAXIMUM_DISTANCE_FOR_POINTING) && (distanceRight<MAXIMUM_DISTANCE_FOR_POINTING) ) { fprintf(stderr,"Cutting off pointing "); return 0; }


  int doHand=1; //1 = right , 2 =left
  if (distanceLeft<distanceRight) { doHand=2; }

  if (doHand==2)
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
   newSkeletonPointingDetected(devID,frameNumber,&skelPF);
   return 1;
  } else
  if (doHand==1)
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
   newSkeletonPointingDetected(devID,frameNumber,&skelPF);
   return 1;
  }

 return 0;
}



void prepareSkeletonState(int devID,unsigned int frameNumber , nite::UserTracker & pUserTracker , const nite::UserData & user  , unsigned int observation , unsigned int totalObservations)
{
    struct skeletonHuman humanSkeleton={0};

    humanSkeleton.observationNumber = observation;
    humanSkeleton.observationTotal = totalObservations;

    humanSkeleton.userID = user.getId();

    humanSkeleton.centerOfMass.x = user.getCenterOfMass().x;
    humanSkeleton.centerOfMass.y = user.getCenterOfMass().y;
    humanSkeleton.centerOfMass.z = user.getCenterOfMass().z;

     nite::SkeletonJoint jointHead           =   user.getSkeleton().getJoint(nite::JOINT_HEAD);
     nite::SkeletonJoint jointNeck           =   user.getSkeleton().getJoint(nite::JOINT_NECK);
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
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_HEAD].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_HEAD].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_HEAD].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_HEAD].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_HEAD].y );
     //------------------------------------------------------------------------------------------



     humanSkeleton.joint[HUMAN_SKELETON_NECK].x = jointNeck.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_NECK].y = jointNeck.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_NECK].z = jointNeck.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_NECK] = jointNeck.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_NECK].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_NECK].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_NECK].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_NECK].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_NECK].y );
     //------------------------------------------------------------------------------------------


     humanSkeleton.joint[HUMAN_SKELETON_LEFT_SHOULDER].x = jointLeftShoulder.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_SHOULDER].y = jointLeftShoulder.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_SHOULDER].z = jointLeftShoulder.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_SHOULDER] = jointLeftShoulder.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_LEFT_SHOULDER].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_SHOULDER].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_SHOULDER].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_SHOULDER].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_SHOULDER].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_SHOULDER].x = jointRightShoulder.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_SHOULDER].y = jointRightShoulder.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_SHOULDER].z = jointRightShoulder.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_SHOULDER] = jointRightShoulder.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_RIGHT_SHOULDER].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_SHOULDER].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_SHOULDER].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_SHOULDER].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_SHOULDER].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_ELBOW].x = jointLeftElbow.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_ELBOW].y = jointLeftElbow.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_ELBOW].z = jointLeftElbow.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_ELBOW] = jointLeftElbow.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_LEFT_ELBOW].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_ELBOW].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_ELBOW].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_ELBOW].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_ELBOW].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_ELBOW].x = jointRightElbow.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_ELBOW].y = jointRightElbow.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_ELBOW].z = jointRightElbow.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_ELBOW] = jointRightElbow.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_RIGHT_ELBOW].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_ELBOW].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_ELBOW].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_ELBOW].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_ELBOW].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HAND].x = jointLeftHand.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HAND].y = jointLeftHand.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HAND].z = jointLeftHand.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_HAND] = jointLeftHand.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_LEFT_HAND].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_HAND].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_HAND].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_HAND].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_HAND].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HAND].x = jointRightHand.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HAND].y = jointRightHand.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HAND].z = jointRightHand.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_HAND] = jointRightHand.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HAND].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HAND].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HAND].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_HAND].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_HAND].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_TORSO].x = jointTorso.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_TORSO].y = jointTorso.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_TORSO].z = jointTorso.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_TORSO] = jointTorso.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_TORSO].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_TORSO].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_TORSO].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_TORSO].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_TORSO].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HIP].x = jointLeftHip.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HIP].y = jointLeftHip.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_HIP].z = jointLeftHip.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_HIP] = jointLeftHip.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_LEFT_HIP].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_HIP].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_HIP].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_HIP].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_HIP].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HIP].x = jointRightHip.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HIP].y = jointRightHip.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HIP].z = jointRightHip.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_HIP] = jointRightHip.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HIP].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HIP].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_HIP].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_HIP].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_HIP].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_KNEE].x = jointLeftKnee.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_KNEE].y = jointLeftKnee.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_KNEE].z = jointLeftKnee.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_KNEE] = jointLeftKnee.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_LEFT_KNEE].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_KNEE].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_KNEE].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_KNEE].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_KNEE].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_KNEE].x = jointRightKnee.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_KNEE].y = jointRightKnee.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_KNEE].z = jointRightKnee.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_KNEE] = jointRightKnee.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_RIGHT_KNEE].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_KNEE].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_KNEE].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_KNEE].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_KNEE].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_LEFT_FOOT].x = jointLeftFoot.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_FOOT].y = jointLeftFoot.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_LEFT_FOOT].z = jointLeftFoot.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_LEFT_FOOT] = jointLeftFoot.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_LEFT_FOOT].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_FOOT].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_LEFT_FOOT].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_FOOT].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_LEFT_FOOT].y );
     //------------------------------------------------------------------------------------------

     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_FOOT].x = jointRightFoot.getPosition().x;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_FOOT].y = jointRightFoot.getPosition().y;
     humanSkeleton.joint[HUMAN_SKELETON_RIGHT_FOOT].z = jointRightFoot.getPosition().z;
     humanSkeleton.jointAccuracy[HUMAN_SKELETON_RIGHT_FOOT] = jointRightFoot.getPositionConfidence();
     pUserTracker.convertJointCoordinatesToDepth(humanSkeleton.joint[HUMAN_SKELETON_RIGHT_FOOT].x ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_FOOT].y ,
                                                 humanSkeleton.joint[HUMAN_SKELETON_RIGHT_FOOT].z ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_FOOT].x ,
                                                 &humanSkeleton.joint2D[HUMAN_SKELETON_RIGHT_FOOT].y );
     //------------------------------------------------------------------------------------------

     //At first take the bounding box given by NiTE ( but this is 2D only and we want a 3D one )
     float maxX=user.getBoundingBox().max.x , maxY=user.getBoundingBox().max.y , maxZ=user.getBoundingBox().max.z;
     float minX=user.getBoundingBox().min.x , minY=user.getBoundingBox().min.y , minZ=user.getBoundingBox().min.z;

     #if CALCULATE_BOUNDING_BOX
     //Use joints to extract bbox
     minX = humanSkeleton.joint[HUMAN_SKELETON_HEAD].x;      maxX = humanSkeleton.joint[HUMAN_SKELETON_HEAD].x;
     minY = humanSkeleton.joint[HUMAN_SKELETON_HEAD].y;      maxY = humanSkeleton.joint[HUMAN_SKELETON_HEAD].y;
     minZ = humanSkeleton.joint[HUMAN_SKELETON_HEAD].z;      maxZ = humanSkeleton.joint[HUMAN_SKELETON_HEAD].z;

     unsigned int i=0;
     for (i=0; i<HUMAN_SKELETON_MIRRORED_PARTS; i++)
      {
        if (humanSkeleton.joint[i].x>maxX) { maxX = humanSkeleton.joint[i].x; } else
        if (humanSkeleton.joint[i].x<minX) { minX = humanSkeleton.joint[i].x; }

        if (humanSkeleton.joint[i].y>maxY) { maxY = humanSkeleton.joint[i].y; } else
        if (humanSkeleton.joint[i].y<minY) { minY = humanSkeleton.joint[i].y; }

        if (humanSkeleton.joint[i].z>maxZ) { maxZ = humanSkeleton.joint[i].z; } else
        if (humanSkeleton.joint[i].z<minZ) { minZ = humanSkeleton.joint[i].z; }
      }
     #endif

     humanSkeleton.bbox[0].x = maxX;       humanSkeleton.bbox[0].y = maxY;   humanSkeleton.bbox[0].z =  minZ;
     humanSkeleton.bbox[1].x = maxX;       humanSkeleton.bbox[1].y = minY;   humanSkeleton.bbox[1].z =  minZ;
     humanSkeleton.bbox[2].x = minX;       humanSkeleton.bbox[2].y = minY;   humanSkeleton.bbox[2].z =  minZ;
     humanSkeleton.bbox[3].x = minX;       humanSkeleton.bbox[3].y = maxY;   humanSkeleton.bbox[3].z =  minZ;
     humanSkeleton.bbox[4].x = maxX;       humanSkeleton.bbox[4].y = maxY;   humanSkeleton.bbox[4].z =  maxZ;
     humanSkeleton.bbox[5].x = maxX;       humanSkeleton.bbox[5].y = minY;   humanSkeleton.bbox[5].z =  maxZ;
     humanSkeleton.bbox[6].x = minX;       humanSkeleton.bbox[6].y = minY;   humanSkeleton.bbox[6].z =  maxZ;
     humanSkeleton.bbox[7].x = minX;       humanSkeleton.bbox[7].y = maxY;   humanSkeleton.bbox[7].z =  maxZ;


  unsigned char statusCalibrating,statusTracking,statusFailed;

    long long unsigned int ts = frameNumber;
	if (user.isNew())  { humanSkeleton.isNew=1; } else
    if ((user.isVisible()) && (!stc[devID].g_visibleUsers[user.getId()])) { humanSkeleton.isVisible=1; }  else
    if ((!user.isVisible()) && (stc[devID].g_visibleUsers[user.getId()])) { humanSkeleton.isOutOfScene=1; } else
    if (user.isLost())  { humanSkeleton.isLost=1; }

	stc[devID].g_visibleUsers[user.getId()] = user.isVisible();
	if(stc[devID].g_skeletonStates[user.getId()] != user.getSkeleton().getState())
	{
		switch(stc[devID].g_skeletonStates[user.getId()] = user.getSkeleton().getState())
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
   newSkeletonDetected(devID,frameNumber,&humanSkeleton);


   if (considerSkeletonPointing(devID,frameNumber,&humanSkeleton))
   {
      fprintf(stderr,"New pointing gesture found\n");
   }

 }




/*

  ------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------
  CODE over here has to do with packing the state of the NITE2 Tracker ( or any other tracker ) to our internal
  format
  ------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------


  ------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------
     Code under here has to do with maintaining the lifecycle of skeleton trackers bound to devices
  ------------------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------------------
*/








int registerSkeletonPointingDetectedEvent(int devID, void * callback)
{
  stc[devID].skelCallbackPointingAddr=callback;
  return 1;
}

int registerSkeletonDetectedEvent(int devID, void * callback)
{
  stc[devID].skelCallbackAddr = callback;
  return 1;
}






int startNite2(int maxVirtualSkeletonTrackers)
{
    printf("Starting Nite2\n");
	nite::NiTE::initialize();
	stc=(struct NiteVirtualDevice * ) malloc(sizeof(struct NiteVirtualDevice) * maxVirtualSkeletonTrackers);
	//memset(stc,0,sizeof(struct NiteVirtualDevice) * maxVirtualSkeletonTrackers);

	int i=0,uid=0;
	for (i=0; i<maxVirtualSkeletonTrackers; i++)
    {
      stc[i].failed=0;
      stc[i].skelCallbackAddr = 0;
      stc[i].skelCallbackPointingAddr = 0;
      for (uid=0; uid<MAX_USERS; uid++)
         {
          stc[uid].g_visibleUsers[uid] = false;
          stc[uid].g_skeletonStates[uid] = nite::SKELETON_NONE;
         }
    }
 return 1;
}


int createNite2Device(int devID,openni::Device * device)
{
    nite::Status niteRc;

    stc[devID].userTracker = new nite::UserTracker;
    stc[devID].userTrackerFrame = new nite::UserTrackerFrameRef;

    if (device==0) {  niteRc = stc[devID].userTracker->create();       } else
	               {  niteRc = stc[devID].userTracker->create(device); }
	if ( niteRc != nite::STATUS_OK)
	{
		fprintf(stderr,RED "Couldn't create user tracker\n" NORMAL);
		if ( niteRc == nite::STATUS_ERROR )       { fprintf(stderr,"Status Error returned\n"); } else
        if ( niteRc == nite::STATUS_BAD_USER_ID ) { fprintf(stderr,"Status Bad User Id returned\n"); } else
        if ( niteRc == nite::STATUS_OUT_OF_FLOW ) { fprintf(stderr,"Status Out of flow returned\n"); }
		stc[devID].failed=1;
		return 0;
	}
	printf("\nStart moving around to get detected...\n(PSI pose may be required for skeleton calibration, depending on the configuration)\n");
	return 1;
}


int destroyNite2Device(int devID)
{
   stc[devID].userTrackerFrame->release();
   
    for (int i=0; i < MAX_USERS; i++) 
   {
		stc[devID].userTracker->stopSkeletonTracking(i+1);
   }
   
   stc[devID].userTracker->destroy();


   delete stc[devID].userTracker;
   delete stc[devID].userTrackerFrame;

   return 1 ;
}



int stopNite2()
{
	nite::NiTE::shutdown();
	#warning "TODO : check if each and every one of stc contexes has been destroyed"
	free ( stc );
	stc=0;
  return 1;
}



int loopNite2(int devID ,unsigned int frameNumber)
{
   if (stc[devID].failed)
     {
       fprintf(stderr,RED "Skeleton Tracking has failed at initialization can't track anything :(\n" NORMAL);
       return 0;
     }

    nite::Status niteRc  = stc[devID].userTracker->readFrame(stc[devID].userTrackerFrame);
	 if (niteRc != nite::STATUS_OK)
		{
			printf("Get next frame failed\n");
			return 0;
		}

		const nite::Array<nite::UserData>& users = stc[devID].userTrackerFrame->getUsers();
		for (int i = 0; i < users.getSize(); ++i)
		{
		    //We will use user from now on as the current user
			const nite::UserData& user = users[i];

            //If our user is new we should start to track his skeleton
			if (user.isNew())
			{
				stc[devID].userTracker->startSkeletonTracking(user.getId());
			}
			else
            //If we have a skeleton tracked , populate our internal structures and call callbacks
            if (user.getSkeleton().getState() == nite::SKELETON_CALIBRATING )
            {
              fprintf(stderr,"Skeleton is beeing calibrated\n");
            }
            if (user.getSkeleton().getState() == nite::SKELETON_TRACKED)
			{
		      prepareSkeletonState(devID,frameNumber,*stc[devID].userTracker,user  , i , users.getSize() );
			}
		}
  return 1;
}



unsigned short  * getNite2DepthFrame(int devID)
{
  return (unsigned short *) stc[devID].userTrackerFrame->getDepthFrame().getData();
}


int getNite2DepthHeight(int devID)
{
	return (int) stc[devID].userTrackerFrame->getDepthFrame().getHeight();
}


int getNite2DepthWidth(int devID)
{
	return (int) stc[devID].userTrackerFrame->getDepthFrame().getWidth();
}