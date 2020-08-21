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


#define USE_TRANSFORM_HASHING 0

struct BVH_Transform
{
  char useOptimizations;
  unsigned char skipCalculationsForJoint[MAX_BVH_JOINT_HIERARCHY_SIZE];
  
  #if USE_TRANSFORM_HASHING
  unsigned int jointIDTransformHashPopulated;
  unsigned int lengthOfListOfJointIDsToTransform;
  BVHJointID listOfJointIDsToTransform[MAX_BVH_JOINT_HIERARCHY_SIZE];
  #endif
  
  struct rectangleArea torso;
  struct BVH_TransformedJoint joint[MAX_BVH_JOINT_HIERARCHY_SIZE];
  float centerPosition[3];
  unsigned int jointsOccludedIn2DProjection;
};





int bvh_populateRectangle2DFromProjections(
                                           struct BVH_MotionCapture * mc ,
                                           struct BVH_Transform * bvhTransform,
                                           struct rectangleArea * area
                                          );


int bvh_printNotSkippedJoints(struct BVH_MotionCapture * bvhMotion ,struct BVH_Transform * bvhTransform);

unsigned char bvh_shouldJointBeTransformedGivenOurOptimizations(const struct BVH_Transform * bvhTransform,const BVHJointID jID);

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
                               //,float * rotationOffset
                             );

int bvh_removeTranslationFromTransform(
                                       struct BVH_MotionCapture * bvhMotion ,
                                       struct BVH_Transform * bvhTransform
                                      );



#ifdef __cplusplus
}
#endif



#endif // BVH_TRANSFORM_H_INCLUDED
