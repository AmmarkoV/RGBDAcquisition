#include <stdio.h>
#include <math.h>
#include "bvh_transform.h"

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"

#define USE_BVH_SPECIFIC_ROTATIONS 0

//Also find Center of Joint
//We can skip the matrix multiplication by just grabbing the last column..
#define FIND_FAST_CENTER 1

//Also find Center of Joint
//We can skip the matrix multiplication by just grabbing the last column..
#define FAST_OFFSET_TRANSLATION 1


#if USE_BVH_SPECIFIC_ROTATIONS
/*
   0  1  2  3
   4  5  6  7
   8  9 10 11
  12 13 14 15
*/

/*
   [0,0]  [1,0]  [2,0]  [3,0]
   [0,1]  [1,1]  [2,1]  [3,1]
   [0,2]  [1,2]  [2,2]  [3,2]
   [0,3]  [1,3]  [2,3]  [3,3]
*/
//---------------------------------------------------------
double degrees_to_radBVH(double degrees)
{
    return (double) degrees * ( (double)  M_PI /180.0);
}
//---------------------------------------------------------
void create4x4RotationBVH_X(double * m,double degrees)
{
    double radians = degrees_to_radBVH(degrees);

    create4x4IdentityMatrix(m);

    double cosV = (double) cosf((float)radians);
    double sinV = (double) sinf((float)radians);

    // Rotate X formula.
    m[5] =    cosV; // [1,1]
    m[9] = -1*sinV; // [1,2]
    m[6] =    sinV; // [2,1]
    m[10] =   cosV; // [2,2]
}
//---------------------------------------------------------
void create4x4RotationBVH_Y(double * m,double degrees)
{
    double radians = degrees_to_radBVH(degrees);

    create4x4IdentityMatrix(m);

    double cosV = (double) cosf((float)radians);
    double sinV = (double) sinf((float)radians);

    // Rotate Y formula.
    m[0] =    cosV; // [0,0]
    m[2] = -1*sinV; // [2,0]
    m[8] =    sinV; // [0,2]
    m[10] =   cosV; // [2,2]
}
//---------------------------------------------------------
void create4x4RotationBVH_Z(double * m,double degrees)
{
    double radians = degrees_to_radBVH(degrees);

    create4x4IdentityMatrix(m);

    double cosV = (double) cosf((float)radians);
    double sinV = (double) sinf((float)radians);

    // Rotate Z formula.
    m[0] =    cosV;  // [0,0]
    m[1] =    sinV;  // [1,0]
    m[4] = -1*sinV;  // [0,1]
    m[5] =    cosV;  // [1,1]
}
//---------------------------------------------------------



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

  if (rotationType==0)
  {
    //No rotation type, get's you back an Identity Matrix..
    return;
  }

//Assuming the rotation axis are correct
//rX,rY,rZ should hold our rotation matrices
   create4x4RotationBVH_X(rX,degreesX);
   create4x4RotationBVH_Y(rY,degreesY);
   create4x4RotationBVH_Z(rZ,degreesZ);


//TODO : Fix correct order..
//http://www.dcs.shef.ac.uk/intranet/research/public/resmes/CS0111.pdf
//https://github.com/duststorm/BVwHacker <- this is a good guide for the transform order..

  switch (rotationType)
  {
    case BVH_ROTATION_ORDER_XYZ :
      //This is what happens with poser exported bvh files..
      //multiplyThree4x4Matrices( matrix, rX, rY, rZ );
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rX);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rY);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rZ);
    break;
    case BVH_ROTATION_ORDER_XZY :
      //multiplyThree4x4Matrices( matrix, rX, rZ, rY );
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rX);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rZ);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rY);
    break;
    case BVH_ROTATION_ORDER_YXZ :
      //multiplyThree4x4Matrices( matrix, rY, rX, rZ );
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rY);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rX);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rZ);
    break;
    case BVH_ROTATION_ORDER_YZX :
      //multiplyThree4x4Matrices( matrix, rY, rZ, rX );
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rY);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rZ);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rX);
    break;
    case BVH_ROTATION_ORDER_ZXY :
      //This is what happens most of the time with bvh files..
      //multiplyThree4x4Matrices( matrix, rZ, rX, rY );
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rZ);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rX);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rY);
    break;
    case BVH_ROTATION_ORDER_ZYX :
      //multiplyThree4x4Matrices( matrix, rZ, rY, rX );
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rZ);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rY);
      multiplyTwo4x4MatricesBuffered(matrix,matrix,rX);
    break;
    default :
      fprintf(stderr,"Error, Incorrect rotation type %u\n",rotationType);
    break;
  };

}
#endif // USE_BVH_SPECIFIC_ROTATIONS


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

  if (fID==0)
  {
   for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
     //---
      create4x4IdentityMatrix(bvhTransform->joint[jID].chainTransformation);
      create4x4IdentityMatrix(bvhTransform->joint[jID].localToWorldTransformation);
     //---
      create4x4IdentityMatrix(bvhTransform->joint[jID].dynamicTranslation);
      create4x4IdentityMatrix(bvhTransform->joint[jID].dynamicRotation);
     //---
   }
  }

  //We need some space to store values
  double posX=0.0,posY=0.0,posZ=0.0;
  double rotX=0.0,rotY=0.0,rotZ=0.0;
  float data[8]={0};

  //First of all we need to populate all local transformation in our chain
  //-----------------------------------------------------------------------
  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
  {
      //Setup dynamic transformation
      //Get values from our bvhMotion structure
      if (!bhv_populatePosXYZRotXYZ(bvhMotion,jID,fID,data,sizeof(data)))
      {
        fprintf(stderr,"Error extracting dynamic transformation for jID=%u @ fID=%u\n",jID,fID);
      }

      posX = fToD(data[0]);
      posY = fToD(data[1]);
      posZ = fToD(data[2]);
      rotX = fToD(data[3]);
      rotY = fToD(data[4]);
      rotZ = fToD(data[5]);

      create4x4TranslationMatrix(bvhTransform->joint[jID].dynamicTranslation,posX,posY,posZ);


     if ( (bvhMotion->jointHierarchy[jID].channelRotationOrder==0)  )
     {
        if (!bvhMotion->jointHierarchy[jID].isEndSite)
              { fprintf(stderr,"No channel rotation order for joint jID=%u jointName=%s, using identity matrix\n",jID,bvhMotion->jointHierarchy[jID].jointName); }
        create4x4IdentityMatrix(bvhTransform->joint[jID].dynamicRotation);
     } else
     {
      #if USE_BVH_SPECIFIC_ROTATIONS
       create4x4RotationBVH(
                            bvhTransform->joint[jID].dynamicRotation,
                            bvhMotion->jointHierarchy[jID].channelRotationOrder,
                            -1*rotX,
                            -1*rotY,
                            -1*rotZ
                           );
      #else
       create4x4MatrixFromEulerAnglesWithRotationOrder(
                                                       bvhTransform->joint[jID].dynamicRotation,
                                                       -1*rotX,
                                                       -1*rotY,
                                                       -1*rotZ,
                                                       (unsigned int) bvhMotion->jointHierarchy[jID].channelRotationOrder
                                                      );
      #endif // USE_BVH_SPECIFIC_ROTATIONS
     }
  }

  //We will now apply all transformations
  //-----------------------------------------------------------------------
  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
  {
     if (bhv_jointHasParent(bvhMotion,jID))
      {
        //If joint is not Root joint
        unsigned int parentID = bvhMotion->jointHierarchy[jID].parentJoint;
        multiplyTwo4x4Matrices(
                                //Output AxB
                                bvhTransform->joint[jID].localToWorldTransformation ,
                                //Parent Output A
                                bvhTransform->joint[parentID].chainTransformation,
                                //This Transform B
                                bvhMotion->jointHierarchy[jID].staticTransformation
                              );
      } else
      if ( bvhMotion->jointHierarchy[jID].isRoot)
      {
       //If we are the root node there is no parent..
       //If there is no parent we will only set our position and copy to the final transform
        #if FAST_OFFSET_TRANSLATION
         create4x4IdentityMatrix(bvhTransform->joint[jID].localToWorldTransformation);
         bvhTransform->joint[jID].localToWorldTransformation[3] =bvhMotion->jointHierarchy[jID].staticTransformation[3]  + bvhTransform->joint[jID].dynamicTranslation[3];
         bvhTransform->joint[jID].localToWorldTransformation[7] =bvhMotion->jointHierarchy[jID].staticTransformation[7]  + bvhTransform->joint[jID].dynamicTranslation[7];
         bvhTransform->joint[jID].localToWorldTransformation[11]=bvhMotion->jointHierarchy[jID].staticTransformation[11] + bvhTransform->joint[jID].dynamicTranslation[11];
         bvhTransform->joint[jID].localToWorldTransformation[15]=1.0;//bvhTransform->joint[jID].staticTransformation[15] + bvhTransform->joint[jID].dynamicTranslation[15];
        #else
         multiplyTwo4x4Matrices(
                                //Output AxB
                                bvhTransform->joint[jID].localToWorldTransformation ,
                                //A
                                bvhMotion->jointHierarchy[jID].staticTransformation,
                                //B
                                bvhTransform->joint[jID].dynamicTranslation
                              );
        #endif // FAST_OFFSET_TRANSLATION
      }


    multiplyTwo4x4Matrices(
                           //Output AxB
                           bvhTransform->joint[jID].chainTransformation ,
                           //A
                           bvhTransform->joint[jID].localToWorldTransformation,
                           //B
                           bvhTransform->joint[jID].dynamicRotation
                          );

  #if FIND_FAST_CENTER
   bvhTransform->joint[jID].pos[0]= bvhTransform->joint[jID].localToWorldTransformation[3];
   bvhTransform->joint[jID].pos[1]= bvhTransform->joint[jID].localToWorldTransformation[7];
   bvhTransform->joint[jID].pos[2]= bvhTransform->joint[jID].localToWorldTransformation[11];
   bvhTransform->joint[jID].pos[3]= bvhTransform->joint[jID].localToWorldTransformation[15];
   normalize3DPointVector(bvhTransform->joint[jID].pos);
  #else
   double centerPoint[4]={0.0,0.0,0.0,1.0};
   transform3DPointVectorUsing4x4Matrix(
                                        bvhTransform->joint[jID].pos,
                                        bvhTransform->joint[jID].localToWorldTransformation,
                                        centerPoint
                                       );
   normalize3DPointVector(bvhTransform->joint[jID].pos);
  #endif // FIND_FAST_CENTER
  }

  return 1;
}


