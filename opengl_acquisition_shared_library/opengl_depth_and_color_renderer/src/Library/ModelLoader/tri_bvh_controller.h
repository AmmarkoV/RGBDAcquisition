/** @file tri_bvh_controller.h
 *  @brief  A module that can transform TRI models from BVH files
 *  @author Ammar Qammaz (AmmarkoV)
 */


#ifndef TRI_BVH_CONTROLLER_H_INCLUDED
#define TRI_BVH_CONTROLLER_H_INCLUDED

#include "model_loader_tri.h"
#include "model_loader_transform_joints.h"
#include "../MotionCaptureLoader/bvh_loader.h"
#include "../MotionCaptureLoader/calculate/bvh_transform.h"

const int animateTRIModelUsingBVHArmature(struct TRI_Model * model,struct BVH_MotionCapture * bvh,unsigned int frameID)
{
 if (model==0) { return 0; }
 if (bvh==0)   { return 0; }
 //--------------------------

  struct BVH_Transform bvhTransform={0};
  if (
       bvh_loadTransformForFrame(
                                  bvh,
                                  frameID,
                                  &bvhTransform,
                                  0
                                )
     )
     {
        fprintf(stderr,"TRI file %s has %u bones\n",model->name,model->header.numberOfBones);
        for (unsigned int boneID=0; boneID<model->header.numberOfBones; boneID++)
        {
         struct TRI_Bones * bone = &model->bones[boneID];
         fprintf(stderr,"Bone %u/%u = %s \n",boneID,model->header.numberOfBones,bone->boneName);
        }

        fprintf(stderr,"BVH file %s has %u jones\n",bvh->fileName,bvh->jointHierarchySize);
        for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
        {
          //bvhTransform->joint[jID].

        }


        return 0;
     } else
     {
       fprintf(stderr,"Error: Failed executing bvh transform\n");
     }
 exit(0);
 return 0;
}




#endif //TRI_BVH_CONTROLLER_H_INCLUDED
