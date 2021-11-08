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


const static void TRIBVH_lowercase(char * str)
{
  char * a = str;
  if (a!=0)
     {
         while (*a!=0)
           {
            *a = tolower(*a);
            ++a;
           }
     }

  return;
}


const static void TRIBVH_removeunderscore(char * str)
{
 char * a = str;
 if (a!=0)
  {
    unsigned int l = strlen(str);
    if (l-2>0)
    {
      if (a[l-2]=='_')
      {
        a[l-2]='.';
      }
    }
  }

  return;
}


const static int removePrefixFromAllTRIBoneNames(struct TRI_Model * triModel,const char * prefix)
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


const static int makeAllTRIBoneNamesLowerCase(struct TRI_Model * triModel)
{
  if (triModel==0) { return 0; }
  //----------------------------------------
  for (unsigned int boneID=0; boneID<triModel->header.numberOfBones; boneID++)
  {
    TRIBVH_lowercase(triModel->bones[boneID].boneName);
  }

  return 1;
}

const static int makeAllTRIBoneNamesLowerCaseWithoutUnderscore(struct TRI_Model * triModel)
{
  if (triModel==0) { return 0; }
  //----------------------------------------
  for (unsigned int boneID=0; boneID<triModel->header.numberOfBones; boneID++)
  {
    char * boneName = triModel->bones[boneID].boneName;
    unsigned int l = strlen(triModel->bones[boneID].boneName);

    TRIBVH_lowercase(triModel->bones[boneID].boneName);

    //These 3 joints need a larget joint name to accommodate the bigger string
    if ( triModel->bones[boneID].boneName ==0 )                     { fprintf(stderr,"Invalid bone name encountered %u \n",boneID);   } else
    if (strcmp(triModel->bones[boneID].boneName,"spine")==0)        {
                                                                      //Allocate enough space for the bone string , read it  , and null terminate it
                                                                      free(triModel->bones[boneID].boneName);
                                                                      triModel->bones[boneID].info->boneNameSize = 8; // 7 + null terminator
                                                                      triModel->bones[boneID].boneName = ( char * ) malloc ( sizeof(char) * (triModel->bones[boneID].info->boneNameSize+1) );
                                                                      boneName = triModel->bones[boneID].boneName;
                                                                      snprintf(boneName,triModel->bones[boneID].info->boneNameSize,"abdomen");
                                                                    }  else
    if (
         (strcmp(triModel->bones[boneID].boneName,"rightarm")==0) ||
         (strcmp(triModel->bones[boneID].boneName,"rshldr")==0)
       )                                                            {
                                                                      //Allocate enough space for the bone string , read it  , and null terminate it
                                                                      free(triModel->bones[boneID].boneName);
                                                                      triModel->bones[boneID].info->boneNameSize = 10; // 9 + null terminator
                                                                      triModel->bones[boneID].boneName = ( char * ) malloc ( sizeof(char) * (triModel->bones[boneID].info->boneNameSize+1) );
                                                                      boneName = triModel->bones[boneID].boneName;
                                                                      snprintf(boneName,triModel->bones[boneID].info->boneNameSize,"rshoulder");
                                                                    }   else
    if (
         (strcmp(triModel->bones[boneID].boneName,"leftarm")==0) ||
         (strcmp(triModel->bones[boneID].boneName,"lshldr")==0)
       )
                                                                    {
                                                                      //Allocate enough space for the bone string , read it  , and null terminate it
                                                                      free(triModel->bones[boneID].boneName);
                                                                      triModel->bones[boneID].info->boneNameSize = 10; // 9 + null terminator
                                                                      triModel->bones[boneID].boneName = ( char * ) malloc ( sizeof(char) * (triModel->bones[boneID].info->boneNameSize+1) );
                                                                      boneName = triModel->bones[boneID].boneName;
                                                                      snprintf(boneName,triModel->bones[boneID].info->boneNameSize,"lshoulder");
                                                                    }   else
    //------------------------------------------------------------------------------------------------------------
    if (strcmp(triModel->bones[boneID].boneName,"hips")==0)         { snprintf(boneName,l,"hip"); }      else
    if (strcmp(triModel->bones[boneID].boneName,"spine1")==0)       { snprintf(boneName,l,"chest"); }    else
    //-------------------------------------------------------------------------------------------------------
    if (strcmp(triModel->bones[boneID].boneName,"rightshoulder")==0){ snprintf(boneName,l,"rcollar"); }  else
    if (
         (strcmp(triModel->bones[boneID].boneName,"rightforearm")==0) ||
         (strcmp(triModel->bones[boneID].boneName,"rforearm")==0)
       )
        { snprintf(boneName,l,"relbow"); }   else
    if (strcmp(triModel->bones[boneID].boneName,"righthand")==0)    { snprintf(boneName,l,"rhand"); }    else
    //-------------------------------------------------------------------------------------------------------
    if (strcmp(triModel->bones[boneID].boneName,"leftshoulder")==0) { snprintf(boneName,l,"lcollar"); }  else
    if (
         (strcmp(triModel->bones[boneID].boneName,"leftforearm")==0) ||
         (strcmp(triModel->bones[boneID].boneName,"lforearm")==0)
       )
         { snprintf(boneName,l,"lelbow"); }   else
    if (strcmp(triModel->bones[boneID].boneName,"lefthand")==0)     { snprintf(boneName,l,"lhand"); }    else
    //-------------------------------------------------------------------------------------------------------
    if (strcmp(triModel->bones[boneID].boneName,"rhipjoint")==0)    { snprintf(boneName,l,"rbuttock"); } else
    if (
         (strcmp(triModel->bones[boneID].boneName,"rightupleg")==0) ||
         (strcmp(triModel->bones[boneID].boneName,"rshin")==0)
       )
         { snprintf(boneName,l,"rhip"); }     else
    if (
         (strcmp(triModel->bones[boneID].boneName,"rightleg")==0) ||
         (strcmp(triModel->bones[boneID].boneName,"rthigh")==0)
       )
         { snprintf(boneName,l,"rknee"); }    else
    if (strcmp(triModel->bones[boneID].boneName,"rightfoot")==0)    { snprintf(boneName,l,"rfoot"); }    else
    //-------------------------------------------------------------------------------------------------------
    if (strcmp(triModel->bones[boneID].boneName,"lhipjoint")==0)    { snprintf(boneName,l,"lbuttock"); } else
    if (
         (strcmp(triModel->bones[boneID].boneName,"leftupleg")==0) ||
         (strcmp(triModel->bones[boneID].boneName,"lshin")==0)
       )
        { snprintf(boneName,l,"lhip"); }     else
    if (
         (strcmp(triModel->bones[boneID].boneName,"leftleg")==0) ||
         (strcmp(triModel->bones[boneID].boneName,"lthigh")==0)
       )
        { snprintf(boneName,l,"lknee"); }    else
    if (strcmp(triModel->bones[boneID].boneName,"leftfoot")==0)     { snprintf(boneName,l,"lfoot"); }

    TRIBVH_removeunderscore(triModel->bones[boneID].boneName);
  }
 return 1;
}





const static int animateTRIModelUsingBVHArmature(
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

        //Cleanup 4x4 matrix transformation..
        for (unsigned int mID=0; mID<numberOfBones; mID++)
        {
            create4x4FIdentityMatrixDirect(&transformations4x4[mID*16]);
        }


        unsigned int * lookupTableFromTRIToBVH = (unsigned int*) malloc(sizeof(unsigned int) * numberOfBones);

        if (lookupTableFromTRIToBVH!=0)
        {
          unsigned int resolvedJoints=0;
          memset(lookupTableFromTRIToBVH,0,sizeof(unsigned int) * numberOfBones);

          for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
           {
              TRIBoneID boneID=0;
              if ( findTRIBoneWithName(modelOriginal,bvh->jointHierarchy[jID].jointName,&boneID) )
              {
                struct TRI_Bones * bone = &modelOriginal->bones[boneID];
                if (printDebugMessages)
                  {
                   fprintf(stderr,GREEN "Resolved BVH Joint %u/%u = `%s`  => ",jID,bvh->jointHierarchySize,bvh->jointHierarchy[jID].jointName);
                   fprintf(stderr,"TRI Bone %u/%u = `%s` \n" NORMAL,boneID,numberOfBones,bone->boneName);
                  }
                lookupTableFromTRIToBVH[boneID]=jID;
                ++resolvedJoints;
              } else
              {
                if (printDebugMessages)
                  { fprintf(stderr,RED "Could not resolve `%s`\n"NORMAL,bvh->jointHierarchy[jID].jointName); }
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
                  //See https://github.com/makehumancommunity/makehuman/blob/master/makehuman/shared/bvh.py#L369

                  memcpy(
                         &transformations4x4[boneID*16], //model.bones[boneID].info->localTransformation,  //localTransformation, //finalVertexTransformation,
                         bvhTransform.joint[jID].dynamicRotation.m, //localToWorldTransformation chainTransformation dynamicRotation dynamicTranslation
                         sizeof(float) * 16
                        );

                 if (bvh->jointHierarchy[jID].hasPositionalChannels)
                 {
                  //This is one of the new joints with positional channels..
                  float * m = &transformations4x4[boneID*16];
                  //m[3]  += ( bvh->jointHierarchy[jID].staticTransformation.m[3] - bvhTransform.joint[jID].dynamicTranslation.m[3]  ) / 10;
                  //m[7]  += ( bvh->jointHierarchy[jID].staticTransformation.m[7] - bvhTransform.joint[jID].dynamicTranslation.m[7]  ) / 10;
                  //m[11] += ( bvh->jointHierarchy[jID].staticTransformation.m[11]- bvhTransform.joint[jID].dynamicTranslation.m[11] ) / 10;

                  struct Matrix4x4OfFloats mergedTranslation;
                  create4x4FTranslationMatrix(
                                              &mergedTranslation,
                                              (bvhTransform.joint[jID].dynamicTranslation.m[3] - bvh->jointHierarchy[jID].staticTransformation.m[3]  ) / 1,
                                              (bvhTransform.joint[jID].dynamicTranslation.m[7] - bvh->jointHierarchy[jID].staticTransformation.m[7]  ) / 1,
                                              (bvhTransform.joint[jID].dynamicTranslation.m[11]- bvh->jointHierarchy[jID].staticTransformation.m[11] ) / 1
                                             );

                  multiplyTwo4x4FMatrices_Naive(
                                                 &transformations4x4[boneID*16],
                                                 bvhTransform.joint[jID].dynamicRotation.m,
                                                 mergedTranslation.m
                                               );
                 }

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
        //---------------------------------------------------------------
        free(transformations4x4);

        return 1;
     } else
     {
       fprintf(stderr,RED "Error: Failed executing bvh transform\n" NORMAL);
     }

 return 0;
}




#endif //TRI_BVH_CONTROLLER_H_INCLUDED
