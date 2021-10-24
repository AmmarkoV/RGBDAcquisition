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


void TRIBVH_removeunderscore(char *a)
{
    if (a==0)
        {
            return;
        }

    unsigned int l = strlen(a);
    if (l-2>0)
    {
      if (a[l-2]=='_')
      {
        a[l-2]='.';
      }
    }
}


int removePrefixFromAllTRIBoneNames(struct TRI_Model * triModel,const char * prefix)
{
  if (triModel==0) { return 0; }
  if (prefix==0)   { return 0; }
  //----------------------------------------
  unsigned int prefixLength = strlen(prefix);

  for (TRIBoneID boneID=0; boneID<triModel->header.numberOfBones; boneID++)
  {
    char * boneName = triModel->bones[boneID].boneName;
    unsigned int fullBoneNameLength = strlen(boneName);

    if ( triModel->bones[boneID].boneName ==0 )
        {
            fprintf(stderr,"Invalid bone name encountered %u \n",boneID);
        } else
        {
          char * result = strstr(triModel->bones[boneID].boneName,prefix);
          if (result!=0)
          {
            snprintf(result,fullBoneNameLength,"%s",boneName+prefixLength);
          }
        }
   }

 return 1;
}


int makeAllTRIBoneNamesLowerCaseWithoutUnderscore(struct TRI_Model * triModel)
{
  if (triModel==0) { return 0; }
  //----------------------------------------
  for (unsigned int boneID=0; boneID<triModel->header.numberOfBones; boneID++)
  {
    char * boneName = triModel->bones[boneID].boneName;
    unsigned int l = strlen(triModel->bones[boneID].boneName);
    unsigned int newStr = strlen(triModel->bones[boneID].boneName);

    //These 3 joints need a larget joint name to accommodate the bigger string
    if ( triModel->bones[boneID].boneName ==0 )                     { fprintf(stderr,"Invalid bone name encountered %u \n",boneID);   } else
    if (strcmp(triModel->bones[boneID].boneName,"Spine")==0)        {
                                                                      //Allocate enough space for the bone string , read it  , and null terminate it
                                                                      free(triModel->bones[boneID].boneName);
                                                                      triModel->bones[boneID].info->boneNameSize = 8; // 7 + null terminator
                                                                      triModel->bones[boneID].boneName = ( char * ) malloc ( sizeof(char) * (triModel->bones[boneID].info->boneNameSize+1) );
                                                                      snprintf(boneName,triModel->bones[boneID].info->boneNameSize,"abdomen");
                                                                    }  else
    if (strcmp(triModel->bones[boneID].boneName,"RightArm")==0)     {
                                                                      //Allocate enough space for the bone string , read it  , and null terminate it
                                                                      free(triModel->bones[boneID].boneName);
                                                                      triModel->bones[boneID].info->boneNameSize = 10; // 9 + null terminator
                                                                      triModel->bones[boneID].boneName = ( char * ) malloc ( sizeof(char) * (triModel->bones[boneID].info->boneNameSize+1) );
                                                                      snprintf(boneName,triModel->bones[boneID].info->boneNameSize,"rshoulder");
                                                                    }   else
    if (strcmp(triModel->bones[boneID].boneName,"LeftArm")==0)      {
                                                                      //Allocate enough space for the bone string , read it  , and null terminate it
                                                                      free(triModel->bones[boneID].boneName);
                                                                      triModel->bones[boneID].info->boneNameSize = 10; // 9 + null terminator
                                                                      triModel->bones[boneID].boneName = ( char * ) malloc ( sizeof(char) * (triModel->bones[boneID].info->boneNameSize+1) );
                                                                      snprintf(boneName,triModel->bones[boneID].info->boneNameSize,"lshoulder");
                                                                    }   else
    //------------------------------------------------------------------------------------------------------------
    if (strcmp(triModel->bones[boneID].boneName,"Hips")==0)         { snprintf(boneName,l,"hip"); }      else
    if (strcmp(triModel->bones[boneID].boneName,"Spine1")==0)       { snprintf(boneName,l,"chest"); }    else
    if (strcmp(triModel->bones[boneID].boneName,"RightShoulder")==0){ snprintf(boneName,l,"rCollar"); }  else
    if (strcmp(triModel->bones[boneID].boneName,"RightForeArm")==0) { snprintf(boneName,l,"relbow"); } else
    if (strcmp(triModel->bones[boneID].boneName,"RightHand")==0)    { snprintf(boneName,l,"rHand"); }    else
    if (strcmp(triModel->bones[boneID].boneName,"LeftShoulder")==0) { snprintf(boneName,l,"lCollar"); }  else
    if (strcmp(triModel->bones[boneID].boneName,"LeftForeArm")==0)  { snprintf(boneName,l,"lelbow"); } else
    if (strcmp(triModel->bones[boneID].boneName,"LeftHand")==0)     { snprintf(boneName,l,"lHand"); }    else
    if (strcmp(triModel->bones[boneID].boneName,"RHipJoint")==0)    { snprintf(boneName,l,"rButtock"); } else
    if (strcmp(triModel->bones[boneID].boneName,"RightUpLeg")==0)   { snprintf(boneName,l,"rhip"); }   else
    if (strcmp(triModel->bones[boneID].boneName,"RightLeg")==0)     { snprintf(boneName,l,"rknee"); }    else
    if (strcmp(triModel->bones[boneID].boneName,"RightFoot")==0)    { snprintf(boneName,l,"rFoot"); }    else
    if (strcmp(triModel->bones[boneID].boneName,"LHipJoint")==0)    { snprintf(boneName,l,"lButtock"); } else
    if (strcmp(triModel->bones[boneID].boneName,"LeftUpLeg")==0)    { snprintf(boneName,l,"lhip"); }   else
    if (strcmp(triModel->bones[boneID].boneName,"LeftLeg")==0)      { snprintf(boneName,l,"lknee"); }    else
    if (strcmp(triModel->bones[boneID].boneName,"LeftFoot")==0)     { snprintf(boneName,l,"lFoot"); }

    TRIBVH_lowercase(triModel->bones[boneID].boneName);
    TRIBVH_removeunderscore(triModel->bones[boneID].boneName);
  }
 return 1;
}





const int animateTRIModelUsingBVHArmature(
                                           struct TRI_Model * modelOutput,
                                           struct TRI_Model * modelOriginal,
                                           struct BVH_MotionCapture * bvh,
                                           unsigned int frameID,
                                           int printDebugMessages
                                         )
{
  if (modelOriginal==0) { return 0; }
  if (modelOutput==0)   { return 0; }
  if (bvh==0)           { return 0; }
  //----------------------------------
  copyModelTri(modelOutput , modelOriginal , 1 /*We also want bone data*/);

  unsigned int numberOfBones = modelOriginal->header.numberOfBones;

  //printTRIBoneStructure(modelOriginal,0 /*alsoPrintMatrices*/);
  //bvh_printBVH(bvh);

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
        unsigned int transformations4x4Size = numberOfBones * 16;
        float * transformations4x4 = (float *) malloc(sizeof(float) * transformations4x4Size);
        if (transformations4x4==0)
        {
            fprintf(stderr,"Failed to allocate enough memory for bones.. \n");
            return 0;
        }

        //memset(transformations4x4,0,sizeof(float) * transformations4x4Size);
        for (unsigned int mID=0; mID<numberOfBones; mID++)
        {
            float * m = &transformations4x4[mID*16];
            //------------------------------------------
            m[0]=1.0;  m[1]=0.0;  m[2]=0.0;   m[3]=0.0;
            m[4]=0.0;  m[5]=1.0;  m[6]=0.0;   m[7]=0.0;
            m[8]=0.0;  m[9]=0.0;  m[10]=1.0;  m[11]=0.0;
            m[12]=0.0; m[13]=0.0; m[14]=0.0;  m[15]=1.0;
        }


        unsigned int * lookupTableFromTRIToBVH = (unsigned int*) malloc(sizeof(unsigned int) * numberOfBones);

        if (lookupTableFromTRIToBVH!=0)
        {
          unsigned int resolvedJoints=0;
          memset(lookupTableFromTRIToBVH,0,sizeof(unsigned int) * numberOfBones);

          if (printDebugMessages)
                  {
                      fprintf(stderr,CYAN "SETUP\n" NORMAL);
                  }

          for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
           {
              TRIBoneID boneID=0;
              if ( findTRIBoneWithName(modelOriginal,bvh->jointHierarchy[jID].jointName,&boneID) )
              {
                struct TRI_Bones * bone = &modelOriginal->bones[boneID];
                if (printDebugMessages)
                  {
                   fprintf(stderr,"Resolved BVH Joint %u/%u = %s  => ",jID,bvh->jointHierarchySize,bvh->jointHierarchy[jID].jointName);
                   fprintf(stderr,"TRI Bone %u/%u = %s \n",boneID,numberOfBones,bone->boneName);
                  }
                lookupTableFromTRIToBVH[boneID]=jID;
                ++resolvedJoints;
              } else
              {
                if (printDebugMessages)
                  { fprintf(stderr,RED "Could not resolve %s\n"NORMAL,bvh->jointHierarchy[jID].jointName); }
              }
           }

          if (resolvedJoints==0)
          {
            if (printDebugMessages)
                  {
                   printTRIBoneStructure(modelOriginal,0 /*alsoPrintMatrices*/);
                   bvh_printBVH(bvh);
                  }
            fprintf(stderr,RED "Could not resolve any joints..!\n" NORMAL);
          } else
          {

          for (unsigned int boneID=0; boneID<numberOfBones; boneID++)
              {
                if (lookupTableFromTRIToBVH[boneID]!=0)
                {
                  BVHJointID jID = lookupTableFromTRIToBVH[boneID];
                  //-----------------------------------------------

                  memcpy(
                         &transformations4x4[boneID*16], //model.bones[boneID].info->localTransformation,  //localTransformation, //finalVertexTransformation,
                         bvhTransform.joint[jID].dynamicRotation.m, //localToWorldTransformation chainTransformation dynamicRotation dynamicTranslation
                         sizeof(float) * 16
                        );

                        /*
                  multiplyTwo4x4FMatrices_Naive(
                                                 &transformations4x4[boneID*16],
                                                 bvhTransform.joint[jID].dynamicRotation.m,
                                                 bvhTransform.joint[jID].dynamicTranslation.m
                                               );*/

                  /*
                  memcpy(
                         &transformations4x4[boneID*16], //model.bones[boneID].info->localTransformation,  //localTransformation, //finalVertexTransformation,
                         bvhTransform.joint[jID].chainTransformation.m, //localToWorldTransformation chainTransformation dynamicRotation dynamicTranslation
                         sizeof(float) * 16
                        );*/

                  //print4x4FMatrix(modelOriginal->bones[boneID].boneName,&transformations4x4[boneID*16],1);
                }
              }
          } //have resolved joints

          //fprintf(stderr,CYAN "resolvedJoints = %u "NORMAL,resolvedJoints);
          free(lookupTableFromTRIToBVH);
        } else
        {
         fprintf(stderr,RED "Error: Could not allocate a lookup table from TRI To BVH\n" NORMAL);
        }


        struct TRI_Model modelTemporary={0};
        //---------------------------------------------------------------
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
        //---------------------------------------------------------------
        fillFlatModelTriFromIndexedModelTri(modelOutput,&modelTemporary);
        deallocInternalsOfModelTri(&modelTemporary);

        free(transformations4x4);

        return 1;
     } else
     {
       fprintf(stderr,RED "Error: Failed executing bvh transform\n" NORMAL);
     }

 return 0;
}




#endif //TRI_BVH_CONTROLLER_H_INCLUDED
