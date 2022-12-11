#ifndef ASSIMP_LOADER_H_INCLUDED
#define ASSIMP_LOADER_H_INCLUDED


#include "model_loader_tri.h"

//#include "../../../../tools/Primitives/skeleton.h"
struct point2D
{
    float x,y;
};

struct point3D
{
    float x,y,z;
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


void deformOriginalModelAndBringBackFlatOneBasedOnThisSkeleton(
                                                                struct TRI_Model * outFlatModel ,
                                                                struct TRI_Model * inOriginalIndexedModel ,
                                                                struct skeletonHuman * sk
                                                              );

int testAssimp(const char * filename  , struct TRI_Model * triModel , struct TRI_Model * originalModel);


#endif // ASSIMP_LOADER_H_INCLUDED
