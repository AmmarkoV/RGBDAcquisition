#ifndef SKELETON_H_INCLUDED
#define SKELETON_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif


static const char * jointNames[] =
{
 "head",
 "neck" ,
 "torso",
 "right_shoulder",
 "left_shoulder",
 "right_elbow",
 "left_elbow",
 "right_hand",
 "left_hand",
 "right_hip",
 "left_hip",
 "right_knee",
 "left_knee",
 "right_foot",
 "left_foot" ,
 "hip" ,
 //=================
 "End of Joint Names"
};


static const char * tgbtNames[] =
{
 "head",
 "neck",
 "bodyCenter",
 "rightShoulder",
 "leftShoulder",
 "rightElbow",
 "leftElbow",
 "rightWrist",
 "leftWrist",
 "rightLegRoot",
 "leftLegRoot",
 "rightKnee",
 "leftKnee",
 "rightAnkle",
 "leftAnkle",
 "hip",
 //--------------------
 "End Of TGBT Names"
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
       "HUMAN_SKELETON_LEFT_FOOT",
       "HUMAN_SKELETON_HIP"
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
   HUMAN_SKELETON_HIP,
   //---------------------
   HUMAN_SKELETON_PARTS
};


static const int humanSkeletonJointsRelationMap[] =
{ // Parent                        Joint
  HUMAN_SKELETON_HEAD,           //HUMAN_SKELETON_HEAD
  HUMAN_SKELETON_HEAD,           //HUMAN_SKELETON_NECK
  HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_TORSO
  HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_RIGHT_SHOULDER
  HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_LEFT_SHOULDER
  HUMAN_SKELETON_RIGHT_SHOULDER, //HUMAN_SKELETON_RIGHT_ELBOW
  HUMAN_SKELETON_LEFT_SHOULDER,  //HUMAN_SKELETON_LEFT_ELBOW
  HUMAN_SKELETON_RIGHT_ELBOW,    //HUMAN_SKELETON_RIGHT_HAND
  HUMAN_SKELETON_LEFT_ELBOW,     //HUMAN_SKELETON_LEFT_HAND
  HUMAN_SKELETON_HIP,            //HUMAN_SKELETON_RIGHT_HIP
  HUMAN_SKELETON_HIP,            //HUMAN_SKELETON_LEFT_HIP
  HUMAN_SKELETON_RIGHT_HIP,      //HUMAN_SKELETON_RIGHT_KNEE
  HUMAN_SKELETON_LEFT_HIP,       //HUMAN_SKELETON_LEFT_KNEE
  HUMAN_SKELETON_LEFT_KNEE,      //HUMAN_SKELETON_RIGHT_FOOT
  HUMAN_SKELETON_LEFT_KNEE,      //HUMAN_SKELETON_LEFT_FOOT
  HUMAN_SKELETON_TORSO,          //HUMAN_SKELETON_HIP
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
      "HUMAN_SKELETON_MIRRORED_HIP"
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
   HUMAN_SKELETON_MIRRORED_HIP,
   //---------------------
   HUMAN_SKELETON_MIRRORED_PARTS
};

   
struct point2D
{
    float x,y;
};

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



static void updateSkeletonBoundingBox(struct skeletonHuman * sk)
{
  //Use joints to extract bbox
  float minX = sk->joint[HUMAN_SKELETON_HEAD].x; float maxX = sk->joint[HUMAN_SKELETON_HEAD].x;
  float minY = sk->joint[HUMAN_SKELETON_HEAD].y; float maxY = sk->joint[HUMAN_SKELETON_HEAD].y;
  float minZ = sk->joint[HUMAN_SKELETON_HEAD].z; float maxZ = sk->joint[HUMAN_SKELETON_HEAD].z;

  unsigned int i=0;
     for (i=0; i<HUMAN_SKELETON_PARTS; i++)
      {
       if (
            (sk->joint[i].x!=0.0) ||
            (sk->joint[i].y!=0.0) ||
            (sk->joint[i].z!=0.0)
           )
        {
         if (sk->joint[i].x>maxX) { maxX = sk->joint[i].x; } else
         if (sk->joint[i].x<minX) { minX = sk->joint[i].x; }

         if (sk->joint[i].y>maxY) { maxY = sk->joint[i].y; } else
         if (sk->joint[i].y<minY) { minY = sk->joint[i].y; }

         if (sk->joint[i].z>maxZ) { maxZ = sk->joint[i].z; } else
         if (sk->joint[i].z<minZ) { minZ = sk->joint[i].z; }
        }
      }

     sk->bbox[0].x = maxX; sk->bbox[0].y = maxY; sk->bbox[0].z = minZ;
     sk->bbox[1].x = maxX; sk->bbox[1].y = minY; sk->bbox[1].z = minZ;
     sk->bbox[2].x = minX; sk->bbox[2].y = minY; sk->bbox[2].z = minZ;
     sk->bbox[3].x = minX; sk->bbox[3].y = maxY; sk->bbox[3].z = minZ;
     sk->bbox[4].x = maxX; sk->bbox[4].y = maxY; sk->bbox[4].z = maxZ;
     sk->bbox[5].x = maxX; sk->bbox[5].y = minY; sk->bbox[5].z = maxZ;
     sk->bbox[6].x = minX; sk->bbox[6].y = minY; sk->bbox[6].z = maxZ;
     sk->bbox[7].x = minX; sk->bbox[7].y = maxY; sk->bbox[7].z = maxZ;

     sk->bboxDimensions.x = (float) sk->bbox[4].x-sk->bbox[2].x;
     sk->bboxDimensions.y = (float) sk->bbox[4].y-sk->bbox[2].y;
     sk->bboxDimensions.z = (float) sk->bbox[4].z-sk->bbox[2].z; 
}


#ifdef __cplusplus
}
#endif

#endif // SKELETON_H_INCLUDED

