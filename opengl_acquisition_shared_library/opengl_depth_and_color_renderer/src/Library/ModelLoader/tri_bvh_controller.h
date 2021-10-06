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


//For lowercase
#include <ctype.h>


void TRIBVH_lowercase(char *a)
{
    if (a==0)
        {
            return;
        }
    while (*a!=0)
        {
            *a = tolower(*a);
            ++a;
        }
}

int makeAllTRIBoneNamesLowerCase(struct TRI_Model * triModel)
{
  for (unsigned int boneID=0; boneID<triModel->header.numberOfBones; boneID++)
  {
    TRIBVH_lowercase(triModel->bones[boneID].boneName);
  }
 return 0;
}





const int animateTRIModelUsingBVHArmature(struct TRI_Model * modelOriginal,struct BVH_MotionCapture * bvh,unsigned int frameID)
{
 if (modelOriginal==0) { return 0; }
 if (bvh==0)   { return 0; }
 //--------------------------

 struct TRI_Model model={0};

 copyModelTri( &model , modelOriginal , 1 /*We also want bone data*/);


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
        fprintf(stderr,"TRI file %s has %u bones\n",model.name,model.header.numberOfBones);
        for (unsigned int boneID=0; boneID<model.header.numberOfBones; boneID++)
        {
         struct TRI_Bones * bone = &model.bones[boneID];
         fprintf(stderr,"TRI Bone %u/%u = %s \n",boneID,model.header.numberOfBones,bone->boneName);
        }

        fprintf(stderr,"BVH file %s has %u joints\n",bvh->fileName,bvh->jointHierarchySize);
        for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
        {
          //bvhTransform->joint[jID].
         fprintf(stderr,"BVH Joint %u/%u = %s \n",jID,bvh->jointHierarchySize,bvh->jointHierarchy[jID].jointName);
        }

        applyVertexTransformation(modelOriginal,&model);

        return 1;
     } else
     {
       fprintf(stderr,"Error: Failed executing bvh transform\n");
     }

 return 0;
}




#endif //TRI_BVH_CONTROLLER_H_INCLUDED
