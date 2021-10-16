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


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


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





const int animateTRIModelUsingBVHArmature(struct TRI_Model * modelOutput,struct TRI_Model * modelOriginal,struct BVH_MotionCapture * bvh,unsigned int frameID)
{
  if (modelOriginal==0) { return 0; }
  if (modelOutput==0)   { return 0; }
  if (bvh==0)           { return 0; }
  //--------------------------

  copyModelTri(modelOutput , modelOriginal , 1 /*We also want bone data*/);

  unsigned int numberOfBones = modelOriginal->header.numberOfBones;

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
        fprintf(stderr,"TRI file %s has %u bones\n",modelOutput->name,numberOfBones);
        for (unsigned int boneID=0; boneID<numberOfBones; boneID++)
        {
         struct TRI_Bones * bone = &modelOriginal->bones[boneID];
         fprintf(stderr,"TRI Bone %u/%u = %s \n",boneID,numberOfBones,bone->boneName);
        }

        fprintf(stderr,"BVH file %s has %u joints\n",bvh->fileName,bvh->jointHierarchySize);
        for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
        {
          //bvhTransform->joint[jID].
         fprintf(stderr,"BVH Joint %u/%u = %s \n",jID,bvh->jointHierarchySize,bvh->jointHierarchy[jID].jointName);
        }

        unsigned int transformations4x4Size = numberOfBones * 16;
        float * transformations4x4 = (float *) malloc(sizeof(float) * transformations4x4Size);
        if (transformations4x4==0)
        {
            fprintf(stderr,"Failed to allocate enough memory for bones.. \n");
            return 0;
        }
        memset(transformations4x4,0,sizeof(float) * transformations4x4Size);


        int * lookupTableFromTRIToBVH = (int*) malloc(sizeof(int) * numberOfBones);

        if (lookupTableFromTRIToBVH!=0)
        {
          memset(lookupTableFromTRIToBVH,0,sizeof(int) * numberOfBones);

          for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
           {
              for (unsigned int boneID=0; boneID<numberOfBones; boneID++)
              {
                struct TRI_Bones * bone = &modelOriginal->bones[boneID];
                if (strcmp(bone->boneName,bvh->jointHierarchy[jID].jointName)==0)
                {
                  fprintf(stderr,"BVH Joint %u/%u = %s  => ",jID,bvh->jointHierarchySize,bvh->jointHierarchy[jID].jointName);
                  fprintf(stderr,"TRI Bone %u/%u = %s \n",boneID,numberOfBones,bone->boneName);
                  lookupTableFromTRIToBVH[boneID]=jID;
                }
              }
           }

          for (unsigned int boneID=0; boneID<numberOfBones; boneID++)
              {
                if (lookupTableFromTRIToBVH[boneID]!=0)
                {
                  BVHJointID jID = lookupTableFromTRIToBVH[boneID];

                  memcpy(
                         &transformations4x4[boneID*16], //model.bones[boneID].info->localTransformation,  //localTransformation, //finalVertexTransformation,
                         bvhTransform.joint[jID].chainTransformation.m,
                         sizeof(float) * 16
                        );
                }
              }

         free(lookupTableFromTRIToBVH);
        }


        struct TRI_Model modelTemporary={0};

        doModelTransform(
                          &modelTemporary ,
                          modelOriginal ,
                          transformations4x4 ,
                          numberOfBones ,
                          1/*Autodetect default matrices for speedup*/ ,
                          1/*Direct setting of matrices*/,
                          1/*Do Transforms, don't just calculate the matrices*/ ,
                          0 /*Default joint convention*/
                        );

        fillFlatModelTriFromIndexedModelTri(modelOutput,&modelTemporary);


        free(transformations4x4);

        return 1;
     } else
     {
       fprintf(stderr,RED "Error: Failed executing bvh transform\n" NORMAL);
     }

 return 0;
}




#endif //TRI_BVH_CONTROLLER_H_INCLUDED
