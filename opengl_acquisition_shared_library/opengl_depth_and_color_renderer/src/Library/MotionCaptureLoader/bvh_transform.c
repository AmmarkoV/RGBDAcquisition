#include <stdio.h>
#include "bvh_transform.h"

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"



//TODO : Fix correct order..
//http://www.dcs.shef.ac.uk/intranet/research/public/resmes/CS0111.pdf
#define FLIP_ROTATION_ORDER 0

#define USE_SCALING_MATRIX 0

//Also find Center of Joint
//We can skip the matrix multiplication by just grabbing the last column..
#define FIND_FAST_CENTER 1



//As http://research.cs.wisc.edu/graphics/Courses/cs-838-1999/Jeff/BVH.html?fbclid=IwAR0Hq96gIhAq-6mvi8OAfMJid2qkv7ZIGNxNMna4vBNngILoceulshvxMfc states
//To calculate the position of a segment you first create a transformation matrix from the local translation and rotation information for that segment. For any joint segment the translation information will simply be the offset as defined in the hierarchy section. The rotation data comes from the motion section. For the root object, the translation data will be the sum of the offset data and the translation data from the motion section. The BVH format doesn't account for scales so it isn't necessary to worry about including a scale factor calculation.
//A straightforward way to create the rotation matrix is to create 3 separate rotation matrices, one for each axis of rotation. Then concatenate the matrices from left to right Y, X and Z.
//
//vR = vYXZ
void create4x4RotationBVH(double * matrix,int rotationType,double degreesX,double degreesY,double degreesZ)
{
  double rX[16]={0};
  double rY[16]={0};
  double rZ[16]={0};

  //Initialize rotation matrix..
  create4x4IdentityMatrix(matrix);

  //Assuming the rotation axis are correct
  //rX,rY,rZ should hold our rotation matrices
  create4x4RotationX(rX,degreesX);
  create4x4RotationX(rY,degreesY);
  create4x4RotationX(rZ,degreesZ);


  #if FLIP_ROTATION_ORDER
  switch (rotationType)
  {
    case BVH_ROTATION_ORDER_XYZ :
      multiplyThree4x4Matrices( matrix, rZ, rY, rX );
    break;
    case BVH_ROTATION_ORDER_XZY :
      multiplyThree4x4Matrices( matrix, rY, rZ, rX );
    break;
    case BVH_ROTATION_ORDER_YXZ :
      multiplyThree4x4Matrices( matrix, rZ, rX, rY );
    break;
    case BVH_ROTATION_ORDER_YZX :
      multiplyThree4x4Matrices( matrix, rX, rZ, rY );
    break;
    case BVH_ROTATION_ORDER_ZXY :
      multiplyThree4x4Matrices( matrix, rY, rX, rZ );
    break;
    case BVH_ROTATION_ORDER_ZYX :
      multiplyThree4x4Matrices( matrix, rX, rY, rZ );
    break;
  };
  #else
  switch (rotationType)
  {
    case BVH_ROTATION_ORDER_XYZ :
      //This is what happens with poser exported bvh files..
      multiplyThree4x4Matrices( matrix, rX, rY, rZ );
    break;
    case BVH_ROTATION_ORDER_XZY :
      multiplyThree4x4Matrices( matrix, rX, rZ, rY );
    break;
    case BVH_ROTATION_ORDER_YXZ :
      multiplyThree4x4Matrices( matrix, rY, rX, rZ );
    break;
    case BVH_ROTATION_ORDER_YZX :
      multiplyThree4x4Matrices( matrix, rY, rZ, rX );
    break;
    case BVH_ROTATION_ORDER_ZXY :
      //This is what happens most of the time with bvh files..
      multiplyThree4x4Matrices( matrix, rZ, rX, rY );
    break;
    case BVH_ROTATION_ORDER_ZYX :
      multiplyThree4x4Matrices( matrix, rZ, rY, rX );
    break;
  };
  #endif // FLIP_ROTATION_ORDER

}


double fToD(float in)
{
  return (double) in;
}

int bvh_loadTransformForFrame(
                               struct BVH_MotionCapture * bvhMotion ,
                               BVHFrameID fID ,
                               struct BVH_Transform * bvhTransform
                             )
{
  unsigned int jID=0;

  //First of all we need to clean the BVH_Transform structure
  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
  {
     create4x4IdentityMatrix(bvhTransform->joint[jID].worldTransformation);
     create4x4IdentityMatrix(bvhTransform->joint[jID].localToWorldTransformation);
     create4x4IdentityMatrix(bvhTransform->joint[jID].staticTransformation);
     create4x4IdentityMatrix(bvhTransform->joint[jID].dynamicTransformation);
     create4x4IdentityMatrix(bvhTransform->joint[jID].dynamicTranslation);
     create4x4IdentityMatrix(bvhTransform->joint[jID].dynamicRotation);
  }

  //We need some space to store the intermediate
  //matrices..
  double translationM[16]={0};
  double rotationM[16]={0};
  double scalingM[16]={0};

  create4x4IdentityMatrix(scalingM);
  create4x4ScalingMatrix(scalingM,0.5,0.5,0.5);

  double posX,posY,posZ;
  double rotX,rotY,rotZ;

  float data[8]={0};



  //First of all we need to populate all local transformation in our chain
  //-----------------------------------------------------------------------
  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
  {
      //Get values from our bvhMotion structure
      bhv_populatePosXYZRotXYZ(bvhMotion,jID,fID,data,sizeof(data));
      float * offset = bvh_getJointOffset(bvhMotion,jID);

      //Setup static transformation
      posX = fToD(offset[0]);
      posY = fToD(offset[1]);
      posZ = fToD(offset[2]);
      create4x4TranslationMatrix(bvhTransform->joint[jID].staticTransformation,posX,posY,posZ);

      //Setip dynamic transformation
      posX = fToD(data[0]);
      posY = fToD(data[1]);
      posZ = fToD(data[2]);
      rotX = fToD(data[3]);
      rotY = fToD(data[4]);
      rotZ = fToD(data[5]);

      create4x4TranslationMatrix(bvhTransform->joint[jID].dynamicTranslation,posX,posY,posZ);
      //create4x4MatrixFromEulerAnglesZYX(rotationM,rotY,rotX,rotZ);
      create4x4RotationBVH(
                            bvhTransform->joint[jID].dynamicRotation,
                            bvhMotion->jointHierarchy[jID].channelRotationOrder,
                            rotX,
                            rotY,
                            rotZ
                          );


      #if USE_SCALING_MATRIX
       multiplyThree4x4Matrices(
                                bvhTransform->joint[jID].dynamicTransformation,
                                bvhTransform->joint[jID].dynamicTranslation,
                                bvhTransform->joint[jID].dynamicRotation,
                                scalingM
                               );
      #else
       multiplyTwo4x4Matrices(
                               bvhTransform->joint[jID].dynamicTransformation,
                               bvhTransform->joint[jID].dynamicTranslation,
                               bvhTransform->joint[jID].dynamicRotation
                            );
      #endif // USE_SCALING_MATRIX
  }


  /*
    if self.parent:
        self.localtoworld = dot(self.parent.trtr, self.stransmat)
    else:
        self.localtoworld = dot(self.stransmat, self.dtransmat)

    # Add rotation of this joint to stack to use for determining children positions
    # Note that position of this joint is not affected by its rotation
    self.trtr = dot(self.localtoworld,self.drotmat)

    # Position is the translation part of the mat (fourth column)
    self.worldpos = array([ self.localtoworld[0,3],
                            self.localtoworld[1,3],
                            self.localtoworld[2,3],
                            self.localtoworld[3,3] ])

  */


  //We will now apply all transformations
  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
  {
     if (bhv_jointHasParent(bvhMotion,jID))
      {//If joint is not Root joint
        unsigned int parentID = bvhMotion->jointHierarchy[jID].parentJoint;
        multiplyTwo4x4Matrices(
                                //Output AxB
                                bvhTransform->joint[jID].localToWorldTransformation ,
                                //Parent Output A
                                bvhTransform->joint[parentID].trtrTransformation,
                                //This Transform B
                                bvhTransform->joint[jID].staticTransformation
                              );

        multiplyTwo4x4Matrices(
                                //Output AxB
                                bvhTransform->joint[jID].trtrTransformation ,
                                //A
                                bvhTransform->joint[jID].localToWorldTransformation,
                                //B
                                bvhTransform->joint[jID].dynamicRotation
                              );

      } else
      {//If we are the root node there is no parent..
       //If there is no parent we will only set our position and copy to the final transform
        multiplyTwo4x4Matrices(
                                //Output AxB
                                bvhTransform->joint[jID].localToWorldTransformation ,
                                //A
                                bvhTransform->joint[jID].staticTransformation,
                                //B
                                bvhTransform->joint[jID].dynamicTranslation
                              );


        multiplyTwo4x4Matrices(
                                //Output AxB
                                bvhTransform->joint[jID].trtrTransformation ,
                                //A
                                bvhTransform->joint[jID].localToWorldTransformation,
                                //B
                                bvhTransform->joint[jID].dynamicRotation
                              );
      }



  #if FIND_FAST_CENTER
   bvhTransform->joint[jID].pos[0]= bvhTransform->joint[jID].localToWorldTransformation[3];
   bvhTransform->joint[jID].pos[1]= bvhTransform->joint[jID].localToWorldTransformation[7];
   bvhTransform->joint[jID].pos[2]= bvhTransform->joint[jID].localToWorldTransformation[11];
   bvhTransform->joint[jID].pos[3]= 1.0;
  #else
   double centerPoint[4]={0.0,0.0,0.0,1.0};
   transform3DPointVectorUsing4x4Matrix(
                                        bvhTransform->joint[jID].pos,
                                        bvhTransform->joint[jID].localToWorldTransformation,
                                        centerPoint
                                       );
  #endif // FIND_FAST_CENTER



  /*
       fprintf(stderr,"Frame %u/Joint : %u \n",fID,jID);
       fprintf(stderr,"Rotation Order : %s \n",rotationOrderNames[bvhMotion->jointHierarchy[jID].channelRotationOrder]);
       print4x4DMatrix(
                        bvhMotion->jointHierarchy[jID].jointName,
                        bvhTransform->joint[jID].localTransformation,
                        1
                      );
  */

  }

  return 1;
}


