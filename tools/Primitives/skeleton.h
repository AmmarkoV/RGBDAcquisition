#ifndef SKELETON_H_INCLUDED
#define SKELETON_H_INCLUDED

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

static float defaultJoints2D[] = { 293.0, 37.0, 293.0, 100.0, 292.0, 170.0, 235.0, 99.0, 349.0, 101.0, 172.0, 144.0, 421.0, 128.0, 113.0, 176.0, 480.0, 144.0, 263.0, 242.0, 319.0, 242.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 291.0, 242.0 };
//static float defaultJoints[] = { -88.01861572265625, -655.5640258789062, 1717.0, -91.15103149414062, -464.8657531738281, 1764.83154296875, -93.02406311035156, -229.4916534423828, 1740.753173828125, -281.5615234375, -465.376708984375, 1757.0, 98.7568359375, -467.6025390625, 1785.0, -498.959716796875, -323.6999206542969, 1784.90966796875, 364.5001220703125, -404.3070373535156, 1923.611572265625, -673.8611450195312, -208.03506469726562, 1731.8184814453125, 562.7467651367188, -339.4126281738281, 1868.3612060546875, -185.67507934570312, 5.257970809936523, 1733.0770263671875, -4.1191253662109375, 6.506894111633301, 1700.2725830078125, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -94.89710235595703, 5.882432460784912, 1716.6748046875 } ;
//static float defaultJoints[] = { -198.40 , -271.72 , 1550.18 , -189.25 , -26.24 , 1621.38 , -192.95 , 152.67 , 1532.26 , -355.50 , -19.27 , 1638.30 , -23.20 , -33.19 , 1604.48 , -481.75 , 233.90 , 1551.29 , 148.30 , 197.92 , 1594.06 , -607.57 , 450.71 , 1439.28 , 310.77 , 401.31 , 1524.37 , -287.40 , 337.91 , 1459.85 , -106.44 , 325.15 , 1426.22 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , -196.92 , 331.53 , 1443.04  } ;


//static float defaultJoints[] = { 95.22 , -206.78 , 894.68 , 100.28 , -7.70 , 1007.84 , 39.10 , 181.37 , 993.07 , -57.30 , -57.94 , 1033.44 , 257.52 , 42.43 , 982.30 , -244.55 , -159.58 , 973.53 , 337.66 , 244.03 , 956.93 , -358.99 , -311.49 , 930.33 , 219.46 , 361.54 , 888.95 , -108.66 , 343.77 , 996.15 , 64.31 , 397.03 , 960.33 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , -22.18 , 370.40 , 978.24 ,  0 } ;
static float defaultJoints[] = { -200.02 , -267.27 , 1554.63 , -197.00 , -20.79 , 1617.80 , -199.95 , 158.42 , 1529.32 , -363.91 , -19.62 , 1622.45 , -29.75 , -21.96 , 1613.14 , -484.64 , 224.27 , 1535.03 , 140.75 , 208.63 , 1609.01 , -606.35 , 452.19 , 1397.94 , 292.78 , 395.32 , 1556.93 , -295.08 , 338.17 , 1445.06 , -110.78 , 337.09 , 1436.65 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , 0.00 , -202.93 , 337.63 , 1440.85 ,  0 } ;


static float defaultAngleOffset[] = { -0.00 , -0.00 , -0.00 , -180.00 , -179.99 , -180.00 , -180.00 , -180.00 , -180.00 , -179.99 , -180.00 , -180.00 , -179.98 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -180.00 , -179.99 , -180.00 , -180.00 , -0.00 , -0.00 , -0.00 , -0.00 , -0.00 , -0.00 , -0.00 , -0.00 , -0.00 , -0.00 , -0.00 , -0.00 , -180.00 , -180.00 , -180.00 ,  0 } ;
static float defaultAngleDirection[] = { 1.00 , 1.00 , 1.00 , 1.00 , 1.0 , 1.00 , 1.00 , 1.00 , 1.00 , 1.0 , 1.00 , 1.00 , 1.0 , 1.0 , 1.0 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.0 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 , 1.00 ,  0 } ;

#define emptyBoundLow -0.01
#define emptyBoundHigh 0.01
#define dp 1700
#define sx 320
#define sy 100
                                      //head         neck         bodycenter   rightshoulder  leftshoulder     rightElbow      leftElbow       rightWrist    leftWrist
static float NAOdefaultJoints2D[] = { sx+0,sy-94 , sx+0,sy+0    , sx+0,sy+22  , sx-191,sy+0 ,  sx+191,sy+0  , sx-191,sy+0   , sx+191,sy+0   , sx-191,sy+0 , sx+191,sy+0 ,
                                      //rightLegRoot  leftLegRoot    rightKnee      leftKnee     rightAnckle      LeftAnckle    hip
                                      sx-80,sy+122 , sx+80,sy+122,  sx-80,sy+200, sx+80,sy+200,  sx-80,sy+300 , sx+80,sy+300 , sx+0,sy+122
                                    };
static float NAOdefaultJoints[] = { 0,-194,dp, 0,0,dp , 0,222,dp , -191,0,dp , 191,0,dp , -191,0,dp-200 , 191,0,dp-200 ,
                                    -191,0,dp-700 , 101,0,dp-700 ,  -80,322,dp , 80,322,dp  , -80,700,dp , 80,700,dp , -80,1000,dp , 80,1000,dp ,
                                    0,322,dp
                                  };


static const char * smartBodyNames[] =
{
    "JtLowerNoseLf",
    "JtNeckB" ,
    "JtSpineB",
    "JtShoulderRt",
    "JtShoulderLf",
    "JtElbowRt",
    "JtElbowLf",
    "JtWristRt",
    "JtWristLf",
    "JtHipRt",
    "JtHipLf",
    "JtKneeRt",
    "JtKneeLf",
    "JtAnkleRt",
    "JtAnkleLf" ,
    "JtSpineA" ,
//=================
    "End of Joint Names" ,
    "Unknown"
};


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
    "End of Joint Names" ,
    "Unknown"
};



static const char * jointNamesFormatted[] =
{
    "Head     ",
    "Neck     " ,
    "Torso    ",
    "RShoulder",
    "LShoulder",
    "RElbow   ",
    "LElbow   ",
    "RHand    ",
    "LHand    ",
    "RHip     ",
    "LHip     ",
    "RKnee    ",
    "LKnee    ",
    "RFoot    ",
    "LFoot    ",
    "Hip      ",
//=================
    "End of Joint Names" ,
    "Unknown"
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
    "End Of TGBT Names" ,
    "Unknown"
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
    "HUMAN_SKELETON_HIP" ,
    "Unknown"
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
    HUMAN_SKELETON_PARTS,
    HUMAN_SKELETON_UNKNOWN
};


static const int humanSkeletonJointsParentRelationMap[] =
{
    // Parent                        Joint
    HUMAN_SKELETON_NECK,           //HUMAN_SKELETON_HEAD
    HUMAN_SKELETON_TORSO,           //HUMAN_SKELETON_NECK
    HUMAN_SKELETON_TORSO,           //HUMAN_SKELETON_TORSO
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
    HUMAN_SKELETON_UNKNOWN
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
    "HUMAN_SKELETON_MIRRORED_HIP" ,
    "Unknown"
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









static const char * COCOBodyNames[] =
{
  "Nose",
  "Neck",
  "RShoulder",
  "RElbow",
  "RWrist",
  "LShoulder",
  "LElbow",
  "LWrist",
  "RHip",
  "RKnee",
  "RAnkle",
  "LHip",
  "LKnee",
  "LAnkle",
  "REye",
  "LEye",
  "REar",
  "LEar",
  "Bkg",
//=================
    "End of Joint Names"
};


enum COCOSkeletonJoints
{
  COCO_Nose,
  COCO_Neck,
  COCO_RShoulder,
  COCO_RElbow,
  COCO_RWrist,
  COCO_LShoulder,
  COCO_LElbow,
  COCO_LWrist,
  COCO_RHip,
  COCO_RKnee,
  COCO_RAnkle,
  COCO_LHip,
  COCO_LKnee,
  COCO_LAnkle,
  COCO_REye,
  COCO_LEye,
  COCO_REar,
  COCO_LEar,
  COCO_Bkg,
   //---------------------
  COCO_PARTS
};

static const int cocoMapToSmartBody[] =
{
    // Values Of Smartbody for a value of coco
    HUMAN_SKELETON_HEAD,
    HUMAN_SKELETON_NECK,
    HUMAN_SKELETON_RIGHT_SHOULDER,
    HUMAN_SKELETON_RIGHT_ELBOW,
    HUMAN_SKELETON_RIGHT_HAND,
    HUMAN_SKELETON_LEFT_SHOULDER,
    HUMAN_SKELETON_LEFT_ELBOW,
    HUMAN_SKELETON_LEFT_HAND,
    HUMAN_SKELETON_RIGHT_HIP,
    HUMAN_SKELETON_RIGHT_KNEE,
    HUMAN_SKELETON_RIGHT_FOOT,
    HUMAN_SKELETON_LEFT_HIP,
    HUMAN_SKELETON_LEFT_KNEE,
    HUMAN_SKELETON_LEFT_FOOT,
    HUMAN_SKELETON_UNKNOWN,
    HUMAN_SKELETON_UNKNOWN,
    HUMAN_SKELETON_UNKNOWN,
    HUMAN_SKELETON_UNKNOWN,
    HUMAN_SKELETON_UNKNOWN,
    HUMAN_SKELETON_UNKNOWN
};


static const int COCOSkeletonJointsParentRelationMap[] =
{
    // Parent                        Joint
  COCO_Nose,                        //COCO_Nose,
  COCO_Nose,                        //COCO_Neck,
  COCO_Neck,                        //COCO_RShoulder,
  COCO_RShoulder,                   //COCO_RElbow,
  COCO_RElbow,                      //COCO_RWrist,
  COCO_Neck,                        //COCO_LShoulder,
  COCO_LShoulder,                   //COCO_LElbow,
  COCO_LElbow,                      //COCO_LWrist,
  COCO_Neck,                        //COCO_RHip,
  COCO_RHip,                        //COCO_RKnee,
  COCO_RKnee,                       //COCO_RAnkle,
  COCO_Neck,                        //COCO_LHip,
  COCO_LHip,                        //COCO_LKnee,
  COCO_LKnee,                       //COCO_LAnkle,
  COCO_Nose,                        //COCO_REye,
  COCO_Nose,                        //COCO_LEye,
  COCO_REye,                        //COCO_REar,
  COCO_LEye,                        //COCO_LEar,
  COCO_Bkg                          //COCO_Bkg
};





struct point2D
{
    float x,y;
};

struct point3D
{
    float x,y,z;
};




enum Point2FlatArray
{
    p_X = 0 ,
    p_Y ,
    p_Z
};


static const char * const Point2FlatArrayNames[] =
{
    "X",
    "Y",
    "Z"
};

struct skeletonNAO
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



struct skeletonCOCO
{
    unsigned int observationNumber , observationTotal;
    unsigned int userID;

    struct point3D bbox[8];

    float  jointAccuracy[COCO_PARTS];
    unsigned int active[COCO_PARTS];
    struct point2D joint2D[COCO_PARTS];
    struct point3D joint[COCO_PARTS];
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
    float  jointAccuracy[HUMAN_SKELETON_PARTS];
    unsigned int active[HUMAN_SKELETON_PARTS];

    struct point3D relativeJointAngle[HUMAN_SKELETON_PARTS];
};


struct skeletonPointing
{
    struct point3D pointStart;
    struct point3D pointEnd;
    struct point3D pointingVector;
    unsigned char isLeftHand;
    unsigned char isRightHand;
};


static int cleanSkeleton(struct skeletonHuman * sk)
{
    unsigned int i=0;

    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        sk->relativeJointAngle[i].x=0.0;
        sk->relativeJointAngle[i].y=0.0;
        sk->relativeJointAngle[i].z=0.0;
        sk->joint[i].x=0.0;
        sk->joint[i].y=0.0;
        sk->joint[i].z=0.0;
        sk->joint2D[i].x=0.0;
        sk->joint2D[i].y=0.0;
        sk->jointAccuracy[i]=0.0;
    }

    return 1;
}


static int skeletonEmpty3DJoint(struct skeletonHuman * sk , unsigned int j1)
{

    //Check NaN
    if (
        (sk->joint[j1].x!=sk->joint[j1].x) ||
        (sk->joint[j1].y!=sk->joint[j1].y) ||
        (sk->joint[j1].z!=sk->joint[j1].z)
    )
    {
        return 1;
    }


    //Check NotSet
    //fprintf(stderr,"%0.2f < %0.2f < %0.2f  X \n ",emptyBoundLow , sk->joint[j1].x , emptyBoundHigh);
    //fprintf(stderr,"%0.2f < %0.2f < %0.2f  Y \n ",emptyBoundLow , sk->joint[j1].y , emptyBoundHigh);
    //fprintf(stderr,"%0.2f < %0.2f < %0.2f  Z \n ",emptyBoundLow , sk->joint[j1].z , emptyBoundHigh);
    if (
        ( (emptyBoundLow < sk->joint[j1].x ) && ( sk->joint[j1].x < emptyBoundHigh) ) &&
        ( (emptyBoundLow < sk->joint[j1].y ) && ( sk->joint[j1].y < emptyBoundHigh) ) &&
        ( (emptyBoundLow < sk->joint[j1].z ) && ( sk->joint[j1].z < emptyBoundHigh) )
    )
    {
        //fprintf(stderr,"Joint %s is empty \n",jointNames[j1]);
        return 1;
    }


    return 0;
}


static int skeletonEmpty2DJoint(struct skeletonHuman * sk , unsigned int j1)
{


    //Check NaN
    if (
        (sk->joint[j1].x!=sk->joint2D[j1].x) ||
        (sk->joint[j1].y!=sk->joint2D[j1].y)
    )
    {
        return 1;
    }

    //Check NotSet
    if (
        (sk->joint2D[j1].x==0) &&
        (sk->joint2D[j1].y==0)
    )
    {
        return 1;
    }


    return 0;
}


static int skeleton3DEmpty(struct skeletonHuman * sk)
{
    unsigned int i=0 , emptyJoints=0;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        if ( skeletonEmpty3DJoint(sk,i) )
        {
            ++emptyJoints;
        }
    }

    if (emptyJoints==HUMAN_SKELETON_PARTS)
    {
        fprintf(stderr,  "All skeleton joints are empty.. \n"   );
        return 1;
    }

    return 0;
}



static int skeleton2DEmpty(struct skeletonHuman * sk)
{
    unsigned int i=0 , emptyJoints=0;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        if ( skeletonEmpty2DJoint(sk,i) )
        {
            ++emptyJoints;
        }
    }

    if (emptyJoints==HUMAN_SKELETON_PARTS)
    {
        fprintf(stderr,  "All skeleton joints are empty.. \n"   );
        return 1;
    }

    return 0;
}


static int skeletonSameJoints(unsigned int j1 , unsigned int j2)
{
    if (humanSkeletonJointsParentRelationMap[j1]==j2)
    {
        return 1;
    }
    return 0;
}



static float skel3Ddistance3D(
                       float * ptA_X , float * ptA_Y , float * ptA_Z ,
                       float * ptB_X , float * ptB_Y , float * ptB_Z
                      )
{
   float sqdiffX = (*ptA_X - *ptB_X);   sqdiffX = sqdiffX * sqdiffX;
   float sqdiffY = (*ptA_Y - *ptB_Y);   sqdiffY = sqdiffY * sqdiffY;
   float sqdiffZ = (*ptA_Z - *ptB_Z);   sqdiffZ = sqdiffZ * sqdiffZ;

   return sqrt( (sqdiffX + sqdiffY + sqdiffZ) );
}


static float skeleton3DGetJointLength(struct skeletonHuman * sk  ,  unsigned int j)
{
  unsigned int pJ =  humanSkeletonJointsParentRelationMap[j];
  if (j!=pJ)
  {
     return skel3Ddistance3D(
                              &sk->joint[j].x,
                              &sk->joint[j].y,
                              &sk->joint[j].z,
                              &sk->joint[pJ].x,
                              &sk->joint[pJ].y,
                              &sk->joint[pJ].z
                            );
  }
 return 0.0;
}




static void updateSkeletonBoundingBox(struct skeletonHuman * sk)
{
    //Use joints to extract bbox
    float minX = sk->joint[HUMAN_SKELETON_HEAD].x;
    float maxX = sk->joint[HUMAN_SKELETON_HEAD].x;
    float minY = sk->joint[HUMAN_SKELETON_HEAD].y;
    float maxY = sk->joint[HUMAN_SKELETON_HEAD].y;
    float minZ = sk->joint[HUMAN_SKELETON_HEAD].z;
    float maxZ = sk->joint[HUMAN_SKELETON_HEAD].z;

    unsigned int i=0;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        if (
            (sk->joint[i].x!=0.0) ||
            (sk->joint[i].y!=0.0) ||
            (sk->joint[i].z!=0.0)
        )
        {
            if (sk->joint[i].x>maxX)
            {
                maxX = sk->joint[i].x;
            }
            else if (sk->joint[i].x<minX)
            {
                minX = sk->joint[i].x;
            }

            if (sk->joint[i].y>maxY)
            {
                maxY = sk->joint[i].y;
            }
            else if (sk->joint[i].y<minY)
            {
                minY = sk->joint[i].y;
            }

            if (sk->joint[i].z>maxZ)
            {
                maxZ = sk->joint[i].z;
            }
            else if (sk->joint[i].z<minZ)
            {
                minZ = sk->joint[i].z;
            }
        }
    }

    sk->bbox[0].x = maxX;
    sk->bbox[0].y = maxY;
    sk->bbox[0].z = minZ;
    sk->bbox[1].x = maxX;
    sk->bbox[1].y = minY;
    sk->bbox[1].z = minZ;
    sk->bbox[2].x = minX;
    sk->bbox[2].y = minY;
    sk->bbox[2].z = minZ;
    sk->bbox[3].x = minX;
    sk->bbox[3].y = maxY;
    sk->bbox[3].z = minZ;
    sk->bbox[4].x = maxX;
    sk->bbox[4].y = maxY;
    sk->bbox[4].z = maxZ;
    sk->bbox[5].x = maxX;
    sk->bbox[5].y = minY;
    sk->bbox[5].z = maxZ;
    sk->bbox[6].x = minX;
    sk->bbox[6].y = minY;
    sk->bbox[6].z = maxZ;
    sk->bbox[7].x = minX;
    sk->bbox[7].y = maxY;
    sk->bbox[7].z = maxZ;

    sk->bboxDimensions.x = (float) sk->bbox[4].x-sk->bbox[2].x;
    sk->bboxDimensions.y = (float) sk->bbox[4].y-sk->bbox[2].y;
    sk->bboxDimensions.z = (float) sk->bbox[4].z-sk->bbox[2].z;
}





static double get2DAngleABCS( double* srcPoint, double* dstObservedPoint, double* dstDefaultPoint )
{
    double *a = dstObservedPoint, *b = srcPoint , *c = dstDefaultPoint;
    /*
                          * dstObservedPoint
                        /          A
                    /
                /
     srcPoint/   Result               C
         B *  -  -  -  -  -  *  dstDefaultPoint

    */

    double ab[2] = { b[0] - a[0], b[1] - a[1] };
    double bc[2] = { c[0] - b[0], c[1] - b[1] };

    double abVec = sqrt(ab[0] * ab[0] + ab[1] * ab[1] );
    double bcVec = sqrt(bc[0] * bc[0] + bc[1] * bc[1] );

    double abNorm[2] = {ab[0] / abVec, ab[1] / abVec };
    double bcNorm[2] = {bc[0] / bcVec, bc[1] / bcVec };

    double res = abNorm[0] * bcNorm[0] + abNorm[1] * bcNorm[1] ;

    res = (double) acos(res)*180.0/ 3.141592653589793;


    /*
     fprintf(stderr,"angle(%0.2f,%0.2f,%0.2f) = %0.2f  [ ab=%0.2f bc=%0.2f abVec=%0.2f bcVec=%0.2f abNorm=%0.2f bcNorm=%0.2f ] ",*srcPoint,*dstPoint,*defaultPoint,res
            ,ab,bc,abVec,bcVec,abNorm,bcNorm
             );
    */

    return res;
}


static int getAngleABCOCV(  double* srcPoint, double* dstObservedPoint, double* dstDefaultPoint )
{
    double *a = dstObservedPoint, *b = srcPoint , *c = dstDefaultPoint;
    /*
                          * dstObservedPoint
                        /          A
                    /
                /
     srcPoint/   Result               C
         B *  -  -  -  -  -  *  dstDefaultPoint

    */
    double ab[2] = { b[0] - a[0], b[1] - a[1] };
    double cb[2] = { c[0] - b[0], c[1] - b[1] };

    float dot = (ab[0] * cb[0] + ab[1] * cb[1]); // dot product
    float cross = (ab[0] * cb[1] - ab[1] * cb[0]); // cross product

    float alpha = atan2(cross, dot);

    return alpha * 180 / M_PI ;
}







static double get2DAngleABC( double* srcPoint, double* dstObservedPoint, double* dstDefaultPoint )
{
    double *a = dstObservedPoint, *b = srcPoint , *c = dstDefaultPoint;
    /*
                          * dstObservedPoint
                        /          A
                    /
                /
     srcPoint/   Result               C
         B *  -  -  -  -  -  *  dstDefaultPoint

    */

    double ab[2] = { a[0] - b[0], a[1] - b[1] };
    double cb[2] = { c[0] - b[0], c[1] - b[1] };

    double abVec = sqrt(ab[0] * ab[0] + ab[1] * ab[1] );
    double cbVec = sqrt(cb[0] * cb[0] + cb[1] * cb[1] );

    double abNorm[2] = {ab[0] / abVec, ab[1] / abVec };
    double cbNorm[2] = {cb[0] / cbVec, cb[1] / cbVec };

    double res = abNorm[0] * cbNorm[0] + abNorm[1] * cbNorm[1] ;

    res = (double) acos(res)*180.0/ 3.141592653589793;


    /*
     fprintf(stderr,"angle(%0.2f,%0.2f,%0.2f) = %0.2f  [ ab=%0.2f bc=%0.2f abVec=%0.2f bcVec=%0.2f abNorm=%0.2f bcNorm=%0.2f ] ",*srcPoint,*dstPoint,*defaultPoint,res
            ,ab,bc,abVec,bcVec,abNorm,bcNorm
             );
    */

    return res;
}


static double getAngleABCRelative(
                                   unsigned int srcPoint , unsigned int dstPoint ,
                                   double* srcObservedPointFirstDimension ,  double* srcObservedPointSecondDimension ,
                                   double* dstObservedPointFirstDimension ,  double* dstObservedPointSecondDimension ,
                                   double* srcDefaultPointFirstDimension  ,  double* srcDefaultPointSecondDimension ,
                                   double* dstDefaultPointFirstDimension  ,  double* dstDefaultPointSecondDimension  ,
                                   unsigned int FirstDimensionConvention  ,
                                   unsigned int SecondDimensionConvention
                                    )
{
    double offsetFirstDim   =  *srcDefaultPointFirstDimension  - *srcObservedPointFirstDimension;
    double offsetSecondDim  =  *srcDefaultPointSecondDimension - *srcObservedPointSecondDimension;

    double relativeDefaultSrc[2] = { *srcDefaultPointFirstDimension                 , *srcDefaultPointSecondDimension };
    double relativeDst[2]        = { *dstObservedPointFirstDimension+offsetFirstDim ,  *dstObservedPointSecondDimension + offsetSecondDim};
    double relativeDefaultDst[2] = { *dstDefaultPointFirstDimension                 ,  *dstDefaultPointSecondDimension  };

    double result=0;

    result = get2DAngleABC(relativeDefaultSrc,relativeDst,relativeDefaultDst);
    //result = getAngleABCOCV(relativeDefaultSrc,relativeDst,relativeDefaultDst);

  if (result>100)
  {
    fprintf(stderr,"getAngleABCRelative%s%s( Parent=%s / Child=%s )\n",Point2FlatArrayNames[FirstDimensionConvention],Point2FlatArrayNames[SecondDimensionConvention],jointNames[srcPoint],jointNames[dstPoint]);

    fprintf(stderr,"%s Observed (%0.2f,%0.2f) - %s Observed (%0.2f,%0.2f) \n",
            jointNames[srcPoint],
            *srcObservedPointFirstDimension,
            *srcObservedPointSecondDimension,

            jointNames[dstPoint],
            *dstObservedPointFirstDimension,
            *dstObservedPointSecondDimension
           );

    fprintf(stderr,"%s Default (%0.2f,%0.2f) - %s Default (%0.2f,%0.2f) \n",
            jointNames[srcPoint],
            *srcDefaultPointFirstDimension,
            *srcDefaultPointSecondDimension,

            jointNames[dstPoint],
            *dstDefaultPointFirstDimension,
            *dstDefaultPointSecondDimension
           );


    fprintf(stderr,"A(%0.2f,%0.2f) B(%0.2f,%0.2f) C(%0.2f,%0.2f) ) = %0.2f \n",
            relativeDefaultSrc[0],
            relativeDefaultSrc[1],
            relativeDst[0],
            relativeDst[1],
            relativeDefaultDst[0],
            relativeDefaultDst[1],
            result);


  }

    return result;
}

static void updateSkeletonAnglesGeneric(struct skeletonHuman * sk , float * defJoints)
{
    unsigned int storeAt=0;
    unsigned int i=0;
    unsigned int src=0,dst=0;

    double srcDA,srcDB , dstDA,dstDB , srcDefDA,srcDefDB , dstDefDA,dstDefDB;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        src = humanSkeletonJointsParentRelationMap[i];
        dst = i;
        storeAt=src;
        sk->active[i]=0;

        if (
                 (!skeletonSameJoints(src,dst)  ) &&
                 (!skeletonEmpty3DJoint(sk,src) ) &&
                 (!skeletonEmpty3DJoint(sk,dst) )
           )
        {
            //We have just observed a skeleton pose , stored at sk->joint and we want to compare it to the defJoints pose
            //So we actually have two points the parent (src) and the child (dst) and we want to calculate the angle there
            //in order to do that
            //We name A , B  the two vectors and we work on a plane every time so

            //Z=A and Y=B gives X rotation
            srcDA = (double) sk->joint[src].z;
            srcDB = (double) sk->joint[src].y;
            dstDA = (double) sk->joint[dst].z;
            dstDB = (double) sk->joint[dst].y;
            srcDefDA = (double) defaultJoints[src*3+p_Z];
            srcDefDB = (double) defaultJoints[src*3+p_Y];
            dstDefDA = (double) defaultJoints[dst*3+p_Z];
            dstDefDB = (double) defaultJoints[dst*3+p_Y];
            sk->relativeJointAngle[storeAt].x=getAngleABCRelative(src,dst, &srcDA,&srcDB,&dstDA,&dstDB,&srcDefDA,&srcDefDB,&dstDefDA,&dstDefDB , p_Z , p_Y);
            //sk->relativeJointAngle[storeAt].x+=defaultAngleOffset[i*3+0];
            //sk->relativeJointAngle[storeAt].x=sk->relativeJointAngle[i].x*defaultAngleDirection[i*3+0];

            //Z and X gives Y
            srcDA = (double) sk->joint[src].z;
            srcDB = (double) sk->joint[src].x;
            dstDA = (double) sk->joint[dst].z;
            dstDB = (double) sk->joint[dst].x;
            srcDefDA = (double) defaultJoints[src*3+p_Z];
            srcDefDB = (double) defaultJoints[src*3+p_X];
            dstDefDA = (double) defaultJoints[dst*3+p_Z];
            dstDefDB = (double) defaultJoints[dst*3+p_X];
            sk->relativeJointAngle[storeAt].y=getAngleABCRelative(src,dst, &srcDA,&srcDB,&dstDA,&dstDB,&srcDefDA,&srcDefDB,&dstDefDA,&dstDefDB , p_Z , p_X);
            //sk->relativeJointAngle[storeAt].y+=defaultAngleOffset[i*3+1];
            //sk->relativeJointAngle[storeAt].y=sk->relativeJointAngle[i].y*defaultAngleDirection[i*3+1];

            //X and Y gives Z
            srcDA = (double) sk->joint[src].x;
            srcDB = (double) sk->joint[src].y;
            dstDA = (double) sk->joint[dst].x;
            dstDB = (double) sk->joint[dst].y;
            srcDefDA = (double) defaultJoints[src*3+p_X];
            srcDefDB = (double) defaultJoints[src*3+p_Y];
            dstDefDA = (double) defaultJoints[dst*3+p_X];
            dstDefDB = (double) defaultJoints[dst*3+p_Y];
            sk->relativeJointAngle[storeAt].z=getAngleABCRelative(src,dst, &srcDA,&srcDB,&dstDA,&dstDB,&srcDefDA,&srcDefDB,&dstDefDA,&dstDefDB , p_X , p_Y);
            //sk->relativeJointAngle[storeAt].z+=defaultAngleOffset[i*3+2];
            //sk->relativeJointAngle[storeAt].z=sk->relativeJointAngle[i].z*defaultAngleDirection[i*3+2];

            unsigned int NaNOutput=0;
            if (sk->relativeJointAngle[i].x!=sk->relativeJointAngle[i].x) { NaNOutput=1; /*sk->relativeJointAngle[i].x=0.0;*/ }
            if (sk->relativeJointAngle[i].y!=sk->relativeJointAngle[i].y) { NaNOutput=1; /*sk->relativeJointAngle[i].y=0.0;*/ }
            if (sk->relativeJointAngle[i].z!=sk->relativeJointAngle[i].z) { NaNOutput=1; /*sk->relativeJointAngle[i].z=0.0;*/ }


            // Check NaN output
            if (!NaNOutput) { sk->active[i]=1; }
        } else

        {
            sk->relativeJointAngle[storeAt].x=0;
            sk->relativeJointAngle[storeAt].y=0;
            sk->relativeJointAngle[storeAt].z=0;
           //fprintf(stderr,"Joint %s->%s combo empty while updating angles \n" , jointNames[src] , jointNames[dst]);
        }

    }

}



static void updateSkeletonAngles(struct skeletonHuman * sk)
{
    updateSkeletonAnglesGeneric( sk , defaultJoints);
}

static void updateSkeletonAnglesNAO(struct skeletonHuman * sk)
{
    updateSkeletonAnglesGeneric( sk , NAOdefaultJoints);
}


static int convertSkeletonFlat3DJointsToPoint3D(struct point3D * out  , float * in )
{
    unsigned int i=0;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        out[i].x = in[i*3+p_X];
        out[i].y = in[i*3+p_Y];
        out[i].z = in[i*3+p_Z];

    }
    return 1;
}



static int convertSkeletonFlat2DJointsToPoint2D(struct point2D * out  , float * in )
{
    unsigned int i=0;
    for (i=0; i<HUMAN_SKELETON_PARTS; i++)
    {
        out[i].x = in[i*2+p_X];
        out[i].y = in[i*2+p_Y];

    }
    return 1;
}


static int fillWithDefaultSkeleton(struct skeletonHuman * sk)
{
    convertSkeletonFlat3DJointsToPoint3D( sk->joint  , defaultJoints );
    convertSkeletonFlat2DJointsToPoint2D( sk->joint2D , defaultJoints2D );
    updateSkeletonAngles(sk);
    return 1;
}


static int fillWithDefaultNAOSkeleton(struct skeletonHuman * sk)
{
    convertSkeletonFlat3DJointsToPoint3D( sk->joint  , NAOdefaultJoints );
    convertSkeletonFlat2DJointsToPoint2D( sk->joint2D , NAOdefaultJoints2D );
    updateSkeletonAngles(sk);
    return 1;
}


static int printSkeletonHuman(struct skeletonHuman * sk)
{
 unsigned int i=0;
 fprintf(stderr," \n");
 fprintf(stderr," __________________________________________________________\n");
 fprintf(stderr,"|SKELETON POSITIONAL DATA - - - - - - - - - - - - - - - - -|\n");
 fprintf(stderr,"|   name            x               y                z     |\n");
 fprintf(stderr,"|__________________________________________________________|\n");
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
          fprintf(stderr,"|");
          if (sk->active[i]) { fprintf(stderr,GREEN " "); } else
                             { fprintf(stderr,RED " ");   }
          fprintf(stderr," %s    %8.2f        %8.2f         %8.2f  ",jointNamesFormatted[i],sk->joint[i].x,sk->joint[i].y,sk->joint[i].z);
          fprintf(stderr,NORMAL"|\n" );
        }
 fprintf(stderr,"|__________________________________________________________|\n\n\n");


 fprintf(stderr," \n");
 fprintf(stderr," __________________________________________________________\n");
 fprintf(stderr,"|SKELETON POSITIONAL DATA DEFAULTS- - - - - - - - - - - - -|\n");
 fprintf(stderr,"|   name            x               y                z     |\n");
 fprintf(stderr,"|__________________________________________________________|\n");
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
          fprintf(stderr,"|");
          if (sk->active[i]) { fprintf(stderr,GREEN " "); } else
                             { fprintf(stderr,RED " ");   }
          fprintf(stderr," %s    %8.2f        %8.2f         %8.2f  ",jointNamesFormatted[i],defaultJoints[i*3+0],defaultJoints[i*3+1],defaultJoints[i*3+2]);
          fprintf(stderr,NORMAL"|\n" );
        }
 fprintf(stderr,"|__________________________________________________________|\n\n\n");


 fprintf(stderr," \n");
 fprintf(stderr," __________________________________________________________\n");
 fprintf(stderr,"|SKELETON POSITIONAL DATA DIFF WITH NEUTRAL POSE- - - - - -|\n");
 fprintf(stderr,"|   name            x               y                z     |\n");
 fprintf(stderr,"|__________________________________________________________|\n");
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
          fprintf(stderr,"|");
          float diffX = sk->joint[i].x-defaultJoints[i*3+0];
          float diffY = sk->joint[i].y-defaultJoints[i*3+1];
          float diffZ = sk->joint[i].z-defaultJoints[i*3+2];
          if ((fabs(diffX)<5) && (fabs(diffY)<5) && (fabs(diffZ)<5) ) { fprintf(stderr,GREEN " "); } else
                                                                      { fprintf(stderr,RED " ");   }


          fprintf(stderr," %s    %8.2f        %8.2f         %8.2f  ",jointNamesFormatted[i],diffX,diffY,diffZ);
          fprintf(stderr,NORMAL"|\n" );
        }
 fprintf(stderr,"|__________________________________________________________|\n\n\n");


 fprintf(stderr," __________________________________________________________\n");
 fprintf(stderr,"|SKELETON ROTATIONAL DATA - - - - - - - - - - - - - - - - -|\n");
 fprintf(stderr,"|   name             x              y                z     |\n");
 fprintf(stderr,"|__________________________________________________________|\n");
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {

          fprintf(stderr,"|");
          if (sk->active[i]) { fprintf(stderr,GREEN " "); } else
                             { fprintf(stderr,RED " ");   }
          fprintf(stderr," %s    %8.2f        %8.2f         %8.2f  ",jointNamesFormatted[i],sk->relativeJointAngle[i].x,sk->relativeJointAngle[i].y,sk->relativeJointAngle[i].z);
          fprintf(stderr,NORMAL"|\n" );

        }
 fprintf(stderr,"|__________________________________________________________|\n");

 fprintf(stderr,"static float defaultJoints[] = { ");
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
          fprintf(stderr,"%0.2f , %0.2f , %0.2f , ",sk->joint[i].x,sk->joint[i].y,sk->joint[i].z);
        }
 fprintf(stderr," 0 } ;\n\n");

 fprintf(stderr,"static float defaultAngleOffset[] = { ");
 for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
          fprintf(stderr,"%0.2f , %0.2f , %0.2f , ",-1*sk->relativeJointAngle[i].x,-1*sk->relativeJointAngle[i].y,-1*sk->relativeJointAngle[i].z);
        }
 fprintf(stderr," 0 } ;\n\n");

return 0;
}

static int visualize2DSkeletonHuman(const char * filename , struct skeletonHuman * sk , float scale)
{
    unsigned int origX=640,origY=480;
    float origXHalf=(float) origX/2,origYHalf=(float) origY/2;


    unsigned int i=0;
    unsigned int src=0,dst=0;
    FILE * fp = fopen(filename,"w");
    if (fp!=0)
    {
        fprintf(fp,"<svg width=\"%0.0f\" height=\"%0.0f\">\n" , origX*scale , origY*scale);

        fprintf(fp,"<text x=\"0\" y=\"20\">\n Timestamp %u\n</text>\n",sk->observationNumber);

        fprintf(fp,"<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" style=\"stroke:rgb(255,255,0);stroke-width:2;stroke-dasharray:10,10\" />\n",
                (float) origXHalf*scale, 0.0 , (float)origXHalf*scale, (float) origY*scale );

        fprintf(fp,"<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" style=\"stroke:rgb(255,255,0);stroke-width:2;stroke-dasharray:10,10\" />\n",
                0.0 ,(float) origYHalf*scale , (float) origX*scale  , (float)origYHalf*scale );


        for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
            src = humanSkeletonJointsParentRelationMap[i];
            dst = i;

            if ( ( !skeletonSameJoints(src,dst) ) && (!skeletonEmpty2DJoint(sk,src)) && (!skeletonEmpty2DJoint(sk,dst)) )
            {
                fprintf(fp,"<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" style=\"stroke:rgb(255,0,0);stroke-width:2\" />\n",
                        sk->joint2D[src].x*scale, sk->joint2D[src].y*scale , sk->joint2D[dst].x*scale, sk->joint2D[dst].y*scale );
            }
        }

        for (i=0; i<HUMAN_SKELETON_PARTS; i++)
        {
            if (!skeletonEmpty2DJoint(sk,i))
            {
                fprintf(fp,"<circle cx=\"%0.2f\" cy=\"%0.2f\" r=\"10\" stroke=\"green\" stroke-width=\"4\" fill=\"yellow\" />\n", sk->joint2D[i].x*scale, sk->joint2D[i].y*scale );


                fprintf(fp,"<text x=\"%0.2f\" y=\"%0.2f\">\n",(sk->joint2D[i].x*scale)-40, (sk->joint2D[i].y*scale)-60   );
                fprintf(fp,"  <tspan fill=\"red\">%s</tspan>",jointNames[i]);
                fprintf(fp,"</text>\n"   );

                fprintf(fp,"<text x=\"%0.2f\" y=\"%0.2f\">\n",(sk->joint2D[i].x*scale)-40, (sk->joint2D[i].y*scale)-40   );
                fprintf(fp,"  <tspan fill=\"red\">%0.2f</tspan>,",sk->joint[i].x);
                fprintf(fp,"  <tspan fill=\"green\">%0.2f</tspan>,",sk->joint[i].y);
                fprintf(fp,"  <tspan fill=\"blue\">%0.2f</tspan>\n",sk->joint[i].z);
                fprintf(fp,"</text>\n"   );

                fprintf(fp,"<text x=\"%0.2f\" y=\"%0.2f\">\n",(sk->joint2D[i].x*scale)-40, (sk->joint2D[i].y*scale)-20   );
                fprintf(fp,"  <tspan fill=\"red\">%0.2f</tspan>/",sk->relativeJointAngle[i].x);
                fprintf(fp,"  <tspan fill=\"green\">%0.2f</tspan>/",sk->relativeJointAngle[i].y);
                fprintf(fp,"  <tspan fill=\"blue\">%0.2f</tspan>\n",sk->relativeJointAngle[i].z);
                fprintf(fp,"</text>\n"   );
            }

        }


        fprintf(fp,"</svg>\n");

        fclose(fp);
        return 1;
    }
    return 0;
}


static int visualize3DSkeletonHuman(const char * filename ,struct skeletonHuman *  sk,int frameNum)
{

  unsigned int i=0;

  fprintf(stderr,YELLOW "printout3DSkeleton(%s,%u)\n" NORMAL,filename,frameNum);

  FILE * fp=0;

  if (frameNum==0)  { fp = fopen(filename,"w"); } else
                    { fp = fopen(filename,"a"); }
  if (fp!=0)
  {
   if (frameNum==0)
   {
   fprintf(fp,"#This is a simple trajectory file..! \n");
   fprintf(fp,"#You can render it with this tool :\n");
   fprintf(fp,"#https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer\n");

   fprintf(fp,"AUTOREFRESH(1500)\n");
   fprintf(fp,"BACKGROUND(20,20,20)\n");
   fprintf(fp,"MOVE_VIEW(1)\n");
   //fprintf(fp,"ALWAYS_SHOW_LAST_FRAME(1)\n");
   fprintf(fp,"INTERPOLATE_TIME(1)\n");
   fprintf(fp,"OBJECTTYPE(joint,sphere)\n");
   fprintf(fp,"OBJECTTYPE(axis3D,axis)\n\n");


   fprintf(fp,"#Bring our world to the MBV coordinate system\n");
   fprintf(fp,"SCALE_WORLD(-0.01,-0.01,0.01)\n");
   fprintf(fp,"MAP_ROTATIONS(-1,-1,1,zxy)\n");
   fprintf(fp,"OFFSET_ROTATIONS(0,0,0)\n");
   fprintf(fp,"EMULATE_PROJECTION_MATRIX(519.460494 , 0.0 , 324.420168 , 0.0 , 519.118667 , 229.823479 , 0 , 1)\n");
   fprintf(fp,"MODELVIEW_MATRIX(1,0,0,0, 0,1,0,0 , 0,0,1,0 ,0,0,0,1)\n");
   fprintf(fp,"#We are now on MBV WORLD !!\n");
   fprintf(fp,"#--------------------------------------------------------------------------\n\n");

   fprintf(fp,"\n#Default Joint Configuration\n");
   for (i=0; i<HUMAN_SKELETON_PARTS; i++)
   {
    fprintf(fp,"OBJECT(def%s,joint,255,0,123,0 ,0, 0.3,0.3,0.3 , )\n",jointNames[i]);
    fprintf(fp,"POS(def%s, 0 ,   %0.2f , %0.2f , %0.2f  , 00.0,0.0,0.0,0.0)\n",jointNames[i],defaultJoints[i*3+0],defaultJoints[i*3+1],defaultJoints[i*3+2]);
   }
   fprintf(fp,"OBJECT(ourAxis,axis3D,255,0,123,0 ,0,1.7,1.7,1.7 , )\n");
    fprintf(fp,"POS(ourAxis, 0 ,  500 , -200 , 1000  , 00.0,0.0,0.0,0.0)\n");
   fprintf(fp,"\n\n");

   fprintf(fp,"\n#Current Joints\n");
   for (i=0; i<HUMAN_SKELETON_PARTS; i++)
   {
    fprintf(fp,"OBJECT(%s,joint,255,255,0,0 ,0, 0.5,0.5,0.5 , )\n",jointNames[i]);
   }

   fprintf(fp,"\n\n");
   for (i=0; i<HUMAN_SKELETON_PARTS; i++)
   {
    fprintf(fp,"CONNECTOR(def%s,%s,0,200,100,0,2.0)\n",jointNames[i],jointNames[i]);
    fprintf(fp,"CONNECTOR(%s,%s,0,0,255,0,3.0)\n",jointNames[i],jointNames[humanSkeletonJointsParentRelationMap[i]]);
    fprintf(fp,"CONNECTOR(def%s,def%s,255,0,123,0,3.0)\n",jointNames[i],jointNames[humanSkeletonJointsParentRelationMap[i]]);
   }

   fprintf(fp,"\n\n");
   }

   fprintf(fp,"POS(camera,%u,   -1.0,1.0, 2.0 , 0.0, 0.0,0.0,0.0 )\n",frameNum*100);

   for (i=0; i<HUMAN_SKELETON_PARTS; i++)
   {
     fprintf(fp,"POS(%s,%u,   %0.2f , %0.2f , %0.2f  , 00.0,0.0,0.0,0.0)\n",jointNames[i],frameNum*100,sk->joint[i].x,sk->joint[i].y,sk->joint[i].z);
   }
  fprintf(fp,"#----------------------- \n\n\n");
  fclose(fp);
  }
 return 0;
}






static int printCOCOSkeletonCSV(struct skeletonCOCO * sk,unsigned int frameNumber)
{
  unsigned int i;
  if (frameNumber==0)
  {
    for (i=0; i<COCO_PARTS-1; i++) { fprintf(stdout,"%s_X,%s_Y,%s_ACC,",COCOBodyNames[i],COCOBodyNames[i],COCOBodyNames[i]); }
    i=COCO_PARTS-1;
    fprintf(stdout,"%s_X,%s_Y,%s_ACC\n",COCOBodyNames[i],COCOBodyNames[i],COCOBodyNames[i]);
  }

  for (i=0; i<COCO_PARTS-1; i++) { fprintf(stdout,"%0.2f,%0.2f,%0.2f,",sk->joint2D[i].x ,sk->joint2D[i].y ,sk->jointAccuracy[i]); }
  i=COCO_PARTS-1;
  fprintf(stdout,"%0.2f,%0.2f,%0.2f\n",sk->joint2D[i].x ,sk->joint2D[i].y ,sk->jointAccuracy[i]);
 return 1;
}

#ifdef __cplusplus
}
#endif

#endif // SKELETON_H_INCLUDED

