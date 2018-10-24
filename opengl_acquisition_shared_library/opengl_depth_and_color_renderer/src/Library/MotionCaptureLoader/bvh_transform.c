#include "bvh_transform.h"

#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"

int bvh_loadTransformForFrame(
                               struct BVH_MotionCapture * bvhMotion ,
                               BVHFrameID fID ,
                               struct BVH_Transform * bvhTransform
                             )
{
  unsigned int jID=0;

  //First of all we need to clean the current transform
  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
  {
     create4x4IdentityMatrix(bvhTransform->joint[jID].localTransformation);
     create4x4IdentityMatrix(bvhTransform->joint[jID].finalVertexTransformation);
  }

  //We will now apply all transformations
  double translationM[16]={0};
  double rotationM[16]={0};
  double scalingM[16]={0};

  create4x4IdentityMatrix(scalingM);
  create4x4ScalingMatrix(scalingM,0.5,0.5,0.5);

  double posX,posY,posZ;
  double rotX,rotY,rotZ;

  float data[8]={0};
  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
  {
      bhv_populatePosXYZRotXYZ(bvhMotion,jID,fID,data,sizeof(data));
      float * offset = bvh_getJointOffset(bvhMotion,jID);

      posX = (double) offset[0] + (double) data[0];
      posY = (double) offset[1] + (double) data[1];
      posZ = (double) offset[2] + (double) data[2];

      rotX = (double) data[3];
      rotY = (double) data[4];
      rotZ = (double) data[5];

      create4x4TranslationMatrix(translationM,posX,posY,posZ);
      create4x4MatrixFromEulerAnglesZYX(rotationM,rotX,rotY,rotZ);

      /*
      multiplyTwo4x4Matrices(
                               bvhTransform->joint[jID].localTransformation,
                               translationM,
                               rotationM
                            );*/

      multiplyThree4x4Matrices(
                               bvhTransform->joint[jID].localTransformation,
                               translationM,
                               rotationM,
                               scalingM
                            );
  }


  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
  {
     if (bhv_jointHasParent(bvhMotion,jID))
      {
        unsigned int parentID = bvhMotion->jointHierarchy[jID].parentJoint;
        multiplyTwo4x4Matrices(
                                bvhTransform->joint[jID].finalVertexTransformation ,
                                bvhTransform->joint[parentID].finalVertexTransformation,
                                bvhTransform->joint[jID].localTransformation
                              );
      } else
      {
        copy4x4DMatrix(
                       bvhTransform->joint[jID].finalVertexTransformation ,
                       bvhTransform->joint[jID].localTransformation
                      );
      }
  }

  return 1;
}


