/** @file bvh_transform.h
 *  @brief  BVH 3D Transormation code.
 *          This part of the code does not deal with loading or changing the BVH file but just performing 3D transformations.
 *          In order to keep things clean all transforms take place in the BVH_Transform structure. The 3D calculation code is
 *          also seperated using the AmMatrix sublibrary https://github.com/AmmarkoV/RGBDAcquisition/tree/master/tools/AmMatrix
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef BVH_TRANSFORM_H_INCLUDED
#define BVH_TRANSFORM_H_INCLUDED

#include "../bvh_loader.h"

#ifdef __cplusplus
extern "C"
{
#endif

/*
   0  1  2  3
   4  5  6  7
   8  9 10 11
  12 13 14 15
*/

struct rectangle3DPointsArea
{
  float x1,y1,z1;
  float x2,y2,z2;
  float x3,y3,z3;
  float x4,y4,z4;
};


struct rectangle2DPointsArea
{
  char calculated;
  float x1,y1;
  float x2,y2;
  float x3,y3;
  float x4,y4;

  float x,y,width,height;
};


struct rectangleArea
{
  char exists,point1Exists,point2Exists,point3Exists,point4Exists;
  int jID[4];
  float averageDepth;
  struct rectangle2DPointsArea rectangle2D;
  struct rectangle3DPointsArea rectangle3D;
};


struct BVH_TransformedJoint
{
  //Flags
  //-----------------
  char pos2DCalculated;
  char isBehindCamera;
  char isOccluded;
  //char skipCalculations;
  char isChainTrasformationComputed;

  //Transforms
  //-----------------
  struct Matrix4x4OfFloats localToWorldTransformation;
  struct Matrix4x4OfFloats chainTransformation;
  struct Matrix4x4OfFloats dynamicTranslation;
  struct Matrix4x4OfFloats dynamicRotation;

  //Position as X,Y,Z
  //-----------------
  float pos3D[4];

  //Position as 2D X,Y
  //-----------------
  float pos2D[2];
};



//Hashing means to generate a new table that only contains
//The needed fields for transforms in order to try skipping
//checks we know that will fail
#define USE_TRANSFORM_HASHING 0

#if USE_TRANSFORM_HASHING
 #warning "Transform hashing is under construction and not yet working.."
#endif


//Transforms should happen on dynamically allocated memory blocks..
#define DYNAMIC_TRANSFORM_ALLOCATIONS 0


//MAX_BVH_JOINT_HIERARCHY_SIZE
#define MAX_BVH_TRANSFORM_SIZE MAX_BVH_JOINT_HIERARCHY_SIZE

struct BVH_Transform
{
  //Memory Management flags..
  char transformStructInitialized;
  unsigned int numberOfJointsSpaceAllocated;
  unsigned int numberOfJointsToTransform;

  //Skip Joint Optimization Logic
  char useOptimizations;

  #if DYNAMIC_TRANSFORM_ALLOCATIONS
   unsigned char * skipCalculationsForJoint;
  #else
   unsigned char skipCalculationsForJoint[MAX_BVH_TRANSFORM_SIZE];
  #endif

  //Transform hashing
  unsigned int jointIDTransformHashPopulated;
  unsigned int lengthOfListOfJointIDsToTransform;
  #if DYNAMIC_TRANSFORM_ALLOCATIONS
   BVHJointID * listOfJointIDsToTransform;
  #else
   BVHJointID listOfJointIDsToTransform[MAX_BVH_TRANSFORM_SIZE];
  #endif

  //Actual Transformation data
  struct rectangleArea torso;
  #if DYNAMIC_TRANSFORM_ALLOCATIONS
   struct BVH_TransformedJoint * joint;
  #else
   struct BVH_TransformedJoint joint[MAX_BVH_TRANSFORM_SIZE];
  #endif
  float centerPosition[3];
  unsigned int jointsOccludedIn2DProjection;
};





int bvh_populateRectangle2DFromProjections(
                                           struct BVH_MotionCapture * mc ,
                                           struct BVH_Transform * bvhTransform,
                                           struct rectangleArea * area
                                          );


void bvh_printBVHTransform(const char * label,struct BVH_MotionCapture * bvhMotion ,struct BVH_Transform * bvhTransform);

void bvh_printNotSkippedJoints(struct BVH_MotionCapture * bvhMotion ,struct BVH_Transform * bvhTransform);

unsigned char bvh_shouldJointBeTransformedGivenOurOptimizations(const struct BVH_Transform * bvhTransform,const BVHJointID jID);

int bvh_markAllJointsAsUsefullInTransform(
                                          struct BVH_MotionCapture * bvhMotion ,
                                          struct BVH_Transform * bvhTransform
                                         );

int bvh_markAllJointsAsUselessInTransform(struct BVH_MotionCapture * bvhMotion,struct BVH_Transform * bvhTransform);

int bvh_markJointAndParentsAsUsefulInTransform(struct BVH_MotionCapture * bvhMotion,struct BVH_Transform * bvhTransform,BVHJointID jID);

int bvh_markJointAsUsefulAndParentsAsUselessInTransform(
                                                        struct BVH_MotionCapture * bvhMotion ,
                                                        struct BVH_Transform * bvhTransform,
                                                        BVHJointID jID
                                                       );
int bvh_markJointAndParentsAsUselessInTransform(
                                                struct BVH_MotionCapture * bvhMotion ,
                                                struct BVH_Transform * bvhTransform,
                                                BVHJointID jID
                                              );

int bvh_loadTransformForMotionBufferFollowingAListOfJointIDs(
                                                             struct BVH_MotionCapture * bvhMotion,
                                                             float * motionBuffer,
                                                             struct BVH_Transform * bvhTransform,
                                                             unsigned int populateTorso,
                                                             BVHJointID * listOfJointIDsToTransform,
                                                             unsigned int lengthOfJointIDList
                                                            );


int bvh_loadTransformForMotionBuffer(
                                     struct BVH_MotionCapture * bvhMotion,
                                     float * motionBuffer,
                                     struct BVH_Transform * bvhTransform,
                                     unsigned int populateTorso
                                   );

int bvh_loadTransformForFrame(
                               struct BVH_MotionCapture * bvhMotion ,
                               BVHFrameID fID ,
                               struct BVH_Transform * bvhTransform,
                               unsigned int populateTorso
                             );

int bvh_removeTranslationFromTransform(
                                       struct BVH_MotionCapture * bvhMotion ,
                                       struct BVH_Transform * bvhTransform
                                      );


int bvh_allocateTransform(struct BVH_MotionCapture * bvhMotion,struct BVH_Transform * bvhTransform);
int bvh_freeTransform(struct BVH_Transform * bvhTransform);

#ifdef __cplusplus
}
#endif



#endif // BVH_TRANSFORM_H_INCLUDED
