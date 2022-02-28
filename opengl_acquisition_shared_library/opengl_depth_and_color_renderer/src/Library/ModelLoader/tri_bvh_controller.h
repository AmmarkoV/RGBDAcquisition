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
#include "../MotionCaptureLoader/edit/bvh_remapangles.h"

#include <stdlib.h> //malloc etc
#include <stdio.h>  //fprintf etc
#include <string.h> //memset etc

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


const static int makeAllTRIBoneNamesLowerCaseWithoutUnderscore(struct TRI_Model * triModel)
{
    if (triModel==0) { return 0; }
    //----------------------------------------
    //----------------------------------------
    //----------------------------------------
    for (unsigned int boneID=0; boneID<triModel->header.numberOfBones; boneID++)
    {
        tri_lowercase(triModel->bones[boneID].boneName);
        tri_removeunderscore(triModel->bones[boneID].boneName);

        //These 3 joints need a larget joint name to accommodate the bigger string
        if ( triModel->bones[boneID].boneName == 0 )
        {
            fprintf(stderr,"Invalid bone name encountered %u \n",boneID);
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"root")==0) ||
             (strcmp(triModel->bones[boneID].boneName,"hips")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"hip");
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"spine1")==0) ||
             (strcmp(triModel->bones[boneID].boneName,"spine01")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"chest");
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"spine")==0) ||
             (strcmp(triModel->bones[boneID].boneName,"spine02")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"abdomen");
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"neck01")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"neck");
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"neck02")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"neck1");
        } else
        //-------------------------------------------------------------------
        if (
            (strcmp(triModel->bones[boneID].boneName,"rightarm")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"rshldr")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"upperarm01.r")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"rshoulder");
        } else
        //-------------------------------------------------------------------
        if (
            (strcmp(triModel->bones[boneID].boneName,"leftarm")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"lshldr")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"upperarm01.l")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"lshoulder");
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"rightshoulder")==0) ||
             (strcmp(triModel->bones[boneID].boneName,"clavicle.r")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"rcollar");
        } else
        //-------------------------------------------------------------------
        if (
            (strcmp(triModel->bones[boneID].boneName,"rightforearm")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"rforearm")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"lowerarm01.r")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"relbow");
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"righthand")==0) ||
             (strcmp(triModel->bones[boneID].boneName,"wrist.r")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"rhand");
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"leftshoulder")==0) ||
             (strcmp(triModel->bones[boneID].boneName,"clavicle.l")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"lcollar");
        } else
        //-------------------------------------------------------------------
        if (
            (strcmp(triModel->bones[boneID].boneName,"leftforearm")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"lforearm")==0)  ||
            (strcmp(triModel->bones[boneID].boneName,"lowerarm01.l")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"lelbow");
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"lefthand")==0) ||
             (strcmp(triModel->bones[boneID].boneName,"wrist.l")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"lhand");
        } else
        //-------------------------------------------------------------------
        if (strcmp(triModel->bones[boneID].boneName,"rhipjoint")==0)
        {
            tri_updateBoneName(triModel,boneID,"rbuttock");
        } else
        //-------------------------------------------------------------------
        if (
            (strcmp(triModel->bones[boneID].boneName,"rightupleg")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"rshin")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"upperleg01.r")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"rhip");
        } else
        //-------------------------------------------------------------------
        if (
            (strcmp(triModel->bones[boneID].boneName,"rightleg")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"rthigh")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"lowerleg01.r")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"rknee");
        } else
        //-------------------------------------------------------------------
        if (
              (strcmp(triModel->bones[boneID].boneName,"rightfoot")==0)  ||
              (strcmp(triModel->bones[boneID].boneName,"foot.r")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"rfoot");
        } else
        //-------------------------------------------------------------------
        if (strcmp(triModel->bones[boneID].boneName,"lhipjoint")==0)
        {
            tri_updateBoneName(triModel,boneID,"lbuttock");
        } else
        //-------------------------------------------------------------------
        if (
            (strcmp(triModel->bones[boneID].boneName,"leftupleg")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"lshin")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"upperleg01.l")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"lhip");
        } else
        //-------------------------------------------------------------------
        if (
            (strcmp(triModel->bones[boneID].boneName,"leftleg")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"lthigh")==0) ||
            (strcmp(triModel->bones[boneID].boneName,"lowerleg01.l")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"lknee");
        } else
        //-------------------------------------------------------------------
        if (
             (strcmp(triModel->bones[boneID].boneName,"leftfoot")==0) ||
             (strcmp(triModel->bones[boneID].boneName,"foot.l")==0)
           )
        {
            tri_updateBoneName(triModel,boneID,"lfoot");
        }
        //-------------------------------------------------------------------------------------------------------

        tri_removeunderscore(triModel->bones[boneID].boneName);
    }
    return 1;
}


/*

r = rotate reference about tail by roll
z = cross(r, tail)
x = cross(tail, z)

Yielding this complete matrix:

/ x.x  tail.x   z.x  head.x \
| x.y  tail.y   z.y  head.y |
| x.z  tail.z   z.z  head.z |
\ 0    0        0    1      /

*/


/*

#
#   M_b = global bone matrix, relative world (PoseBone.matrix)
#   L_b = local bone matrix, relative parent and rest (PoseBone.matrix_local)
#   R_b = bone rest matrix, relative armature (Bone.matrix_local)
#   T_b = global T-pose marix, relative world
#
#   M_b = M_p R_p^-1 R_b L_b
#   M_b = A_b M'_b
#   T_b = A_b T'_b
#   A_b = T_b T'^-1_b
#   B_b = R^-1_b R_p
#
#   L_b = R^-1_b R_p M^-1_p A_b M'_b
#   L_b = B_b M^-1_p A_b M'_b
#



def getRollMat(mat):
    quat = mat.to_3x3().to_quaternion()
    if abs(quat.w) < 1e-4:
        roll = pi
    else:
        roll = -2*math.atan(quat.y/quat.w)
    return roll



def getHeadTailDir(pb):
    mat = pb.bone.matrix_local
    mat = pb.matrix
    head = Vector(mat.col[3][:3])
    vec = Vector(mat.col[1][:3])
    tail = head + pb.bone.length * vec
    return head, tail, vec
*/
const static unsigned int * createLookupTableFromTRItoBVH(
                                                          struct TRI_Model * modelOriginal,
                                                          struct BVH_MotionCapture * bvh,
                                                          int printDebugMessages
                                                         )
{
 if (bvh==0)           { fprintf(stderr,"createLookupTableFromTRItoBVH cannot work without a BVH Model\n"); return 0; }
 if (modelOriginal==0) { fprintf(stderr,"createLookupTableFromTRItoBVH cannot work without a TRI Model\n"); return 0; }
 //--------------------------------------------------------------------------------------------------------------------
 unsigned int resolvedJoints=0;
 unsigned int numberOfBones = modelOriginal->header.numberOfBones;
 //--------------------------------------------------------------------------------------------------------------------
 unsigned int * lookupTableFromTRIToBVH = (unsigned int*) malloc(sizeof(unsigned int) * numberOfBones);
 //--------------------------------------------------------------------------------------------------------------------

 if (lookupTableFromTRIToBVH!=0)
 {
  memset(lookupTableFromTRIToBVH,0,sizeof(unsigned int) * numberOfBones);

  for (BVHJointID jID=0; jID<bvh->jointHierarchySize; jID++)
            {
                TRIBoneID boneID=0;
                if ( tri_findBone(modelOriginal,bvh->jointHierarchy[jID].jointName,&boneID) )
                {
                    struct TRI_Bones * bone = &modelOriginal->bones[boneID];
                    if (printDebugMessages)
                    {
                        fprintf(stderr,GREEN "Resolved BVH Joint %u/%u = `%s`  => ",jID,bvh->jointHierarchySize,bvh->jointHierarchy[jID].jointName);
                        fprintf(stderr,"TRI Bone %u/%u = `%s` \n" NORMAL,boneID,numberOfBones,bone->boneName);
                    }
                    lookupTableFromTRIToBVH[boneID]=jID;
                    ++resolvedJoints;
                }
                else
                {
                    if (printDebugMessages)
                    {
                        fprintf(stderr,RED "Could not resolve `%s`\n"NORMAL,bvh->jointHierarchy[jID].jointName);
                    }
                }
            }

  if (resolvedJoints==0)
            {
                if (printDebugMessages)
                {
                    printTRIBoneStructure(modelOriginal,0 /*alsoPrintMatrices*/);
                    bvh_printBVH(bvh);
                }
                fprintf(stderr,RED "Could not resolve any joints, freeing empty map..!\n" NORMAL);
                free(lookupTableFromTRIToBVH);
                lookupTableFromTRIToBVH = 0;
            } else
            {
                fprintf(stderr,CYAN "Resolved %u joints..!\n" NORMAL,resolvedJoints);
            }
 }

  return lookupTableFromTRIToBVH;
}



struct testResult
{
  BVHMotionChannelID mID;
  float value;
  float dX;
  float dY;
  float dZ;
};





const static int checkBVHRotation(
                                  struct testResult * bvhResult,
                                  struct BVH_MotionCapture * bvh,
                                  const char * bvhJointName
                                 )
{
    BVHJointID childJID=0;
    BVHJointID jID=0;
    if (!bvh_getJointIDFromJointNameNocase(bvh,bvhJointName,&jID))
    {
        fprintf(stderr,"Could not resolve  BVH joint %s \n",bvhJointName);
        return 0;
    }
    for (BVHJointID j=0; j<bvh->jointHierarchySize; j++)
    {
       if (bvh->jointHierarchy[j].parentJoint == jID)
       {
          childJID = j;
          break;
       }
    }


    struct MotionBuffer * frameMotionBuffer = mallocNewMotionBuffer(bvh);

    if (frameMotionBuffer!=0)
       {
        struct BVH_Transform bvhTransform = {0};
        bvhTransform.useOptimizations=0;
        if (
             !bvh_loadTransformForMotionBuffer(
                                                bvh,
                                                frameMotionBuffer->motion,
                                                &bvhTransform,
                                                0
                                              )
           )
        {
            freeMotionBuffer(&frameMotionBuffer);
            fprintf(stderr,"Could not do BVH Transform loading\n");
            return 0;
        }


         BVHMotionChannelID mID = 0;


         //All zeros..!
         unsigned int testID = 0;
         bvh_loadTransformForMotionBuffer(bvh,frameMotionBuffer->motion,&bvhTransform,0);
         bvhResult[testID].dX = (float) bvhTransform.joint[jID].pos3D[0] - bvhTransform.joint[childJID].pos3D[0];
         bvhResult[testID].dY = (float) bvhTransform.joint[jID].pos3D[1] - bvhTransform.joint[childJID].pos3D[1];
         bvhResult[testID].dZ = (float) bvhTransform.joint[jID].pos3D[2] - bvhTransform.joint[childJID].pos3D[2];


         // Z
         //-----------------------------------------------------------------------------------------------
         mID = bvh->jointToMotionLookup[jID].channelIDMotionOffset[BVH_ROTATION_Z];
         //mID = bvh->jointToMotionLookup[jID].jointMotionOffset;
         ++testID;
         frameMotionBuffer->motion[mID]=-90.0;
         bvh_loadTransformForMotionBuffer(bvh,frameMotionBuffer->motion,&bvhTransform,0);
         bvhResult[testID].dX = (float) bvhTransform.joint[jID].pos3D[0] - bvhTransform.joint[childJID].pos3D[0];
         bvhResult[testID].dY = (float) bvhTransform.joint[jID].pos3D[1] - bvhTransform.joint[childJID].pos3D[1];
         bvhResult[testID].dZ = (float) bvhTransform.joint[jID].pos3D[2] - bvhTransform.joint[childJID].pos3D[2];
         bvhResult[testID].value = frameMotionBuffer->motion[mID];
         bvhResult[testID].mID   = mID;

         ++testID;
         frameMotionBuffer->motion[mID]=+90.0;
         bvh_loadTransformForMotionBuffer(bvh,frameMotionBuffer->motion,&bvhTransform,0);
         bvhResult[testID].dX = (float) bvhTransform.joint[jID].pos3D[0] - bvhTransform.joint[childJID].pos3D[0];
         bvhResult[testID].dY = (float) bvhTransform.joint[jID].pos3D[1] - bvhTransform.joint[childJID].pos3D[1];
         bvhResult[testID].dZ = (float) bvhTransform.joint[jID].pos3D[2] - bvhTransform.joint[childJID].pos3D[2];
         bvhResult[testID].value = frameMotionBuffer->motion[mID];
         bvhResult[testID].mID   = mID;
         frameMotionBuffer->motion[mID]=0.0;
         //-----------------------------------------------------------------------------------------------


         // X
         //-----------------------------------------------------------------------------------------------
         mID = bvh->jointToMotionLookup[jID].channelIDMotionOffset[BVH_ROTATION_X];
         //mID = bvh->jointToMotionLookup[jID].jointMotionOffset+1;
         ++testID;
         frameMotionBuffer->motion[mID]=-90.0;
         bvh_loadTransformForMotionBuffer(bvh,frameMotionBuffer->motion,&bvhTransform,0);
         bvhResult[testID].dX = (float) bvhTransform.joint[jID].pos3D[0] - bvhTransform.joint[childJID].pos3D[0];
         bvhResult[testID].dY = (float) bvhTransform.joint[jID].pos3D[1] - bvhTransform.joint[childJID].pos3D[1];
         bvhResult[testID].dZ = (float) bvhTransform.joint[jID].pos3D[2] - bvhTransform.joint[childJID].pos3D[2];
         bvhResult[testID].value = frameMotionBuffer->motion[mID];
         bvhResult[testID].mID   = mID;

         ++testID;
         frameMotionBuffer->motion[mID]=+90.0;
         bvh_loadTransformForMotionBuffer(bvh,frameMotionBuffer->motion,&bvhTransform,0);
         bvhResult[testID].dX = (float) bvhTransform.joint[jID].pos3D[0] - bvhTransform.joint[childJID].pos3D[0];
         bvhResult[testID].dY = (float) bvhTransform.joint[jID].pos3D[1] - bvhTransform.joint[childJID].pos3D[1];
         bvhResult[testID].dZ = (float) bvhTransform.joint[jID].pos3D[2] - bvhTransform.joint[childJID].pos3D[2];
         bvhResult[testID].value = frameMotionBuffer->motion[mID];
         bvhResult[testID].mID   = mID;
         frameMotionBuffer->motion[mID]=0.0;
         //-----------------------------------------------------------------------------------------------

         // Y
         //-----------------------------------------------------------------------------------------------
         mID = bvh->jointToMotionLookup[jID].channelIDMotionOffset[BVH_ROTATION_Y];
         //mID = bvh->jointToMotionLookup[jID].jointMotionOffset+2;
         ++testID;
         frameMotionBuffer->motion[mID]=-90;
         bvh_loadTransformForMotionBuffer(bvh,frameMotionBuffer->motion,&bvhTransform,0);
         bvhResult[testID].dX = (float) bvhTransform.joint[jID].pos3D[0] - bvhTransform.joint[childJID].pos3D[0];
         bvhResult[testID].dY = (float) bvhTransform.joint[jID].pos3D[1] - bvhTransform.joint[childJID].pos3D[1];
         bvhResult[testID].dZ = (float) bvhTransform.joint[jID].pos3D[2] - bvhTransform.joint[childJID].pos3D[2];
         bvhResult[testID].value = frameMotionBuffer->motion[mID];
         bvhResult[testID].mID   = mID;

         ++testID;
         frameMotionBuffer->motion[mID]=+90;
         bvh_loadTransformForMotionBuffer(bvh,frameMotionBuffer->motion,&bvhTransform,0);
         bvhResult[testID].dX = (float) bvhTransform.joint[jID].pos3D[0] - bvhTransform.joint[childJID].pos3D[0];
         bvhResult[testID].dY = (float) bvhTransform.joint[jID].pos3D[1] - bvhTransform.joint[childJID].pos3D[1];
         bvhResult[testID].dZ = (float) bvhTransform.joint[jID].pos3D[2] - bvhTransform.joint[childJID].pos3D[2];
         bvhResult[testID].value = frameMotionBuffer->motion[mID];
         bvhResult[testID].mID   = mID;
         frameMotionBuffer->motion[mID]=0.0;
         //-----------------------------------------------------------------------------------------------


         for (int i=0; i<7; i++)
         {
            fprintf(stderr,"Test %u (mID=%u set to %0.2f) ",i,bvhResult[i].mID,bvhResult[i].value);
            if (bvhResult[i].value>0.0) { fprintf(stderr," "); }
            fprintf(stderr," | ");

            //--------------------------------------------------------------
            if (bvhResult[i].dX>0.0)  {  fprintf(stderr,"+ ");  bvhResult[i].dX= 1.0; } else
            if (bvhResult[i].dX<0.0)  {  fprintf(stderr,"- ");  bvhResult[i].dX=-1.0; } else
                                      {  fprintf(stderr,"0 ");  bvhResult[i].dX= 0.0; }
            //--------------------------------------------------------------
            if (bvhResult[i].dY>0.0)  {  fprintf(stderr,"+ ");  bvhResult[i].dY= 1.0; } else
            if (bvhResult[i].dY<0.0)  {  fprintf(stderr,"- ");  bvhResult[i].dY=-1.0; } else
                                      {  fprintf(stderr,"0 ");  bvhResult[i].dY= 0.0; }
            //--------------------------------------------------------------
            if (bvhResult[i].dZ>0.0)  {  fprintf(stderr,"+ ");  bvhResult[i].dZ= 1.0; } else
            if (bvhResult[i].dZ<0.0)  {  fprintf(stderr,"- ");  bvhResult[i].dZ=-1.0; } else
                                      {  fprintf(stderr,"0 ");  bvhResult[i].dZ= 0.0; }
            //--------------------------------------------------------------
            fprintf(stderr,"\n");
         }


        freeMotionBuffer(&frameMotionBuffer);
        return 1;
       }

   return 0;
}





const static int checkTRIRotation(
                                  struct testResult * triResult,
                                  struct TRI_Model * modelOriginal,
                                  const char * triJointName,
                                  int childOfChild
                                 )
{
    if (modelOriginal==0)
    {
         fprintf(stderr,"No TRI model ?\n");
         return 0;
    }
    if (modelOriginal->bones==0)
    {
         fprintf(stderr,"No bones ?\n");
         return 0;
    }


    unsigned int boneChildID=0;
    unsigned int boneID=0;
    if (!tri_findBone(modelOriginal,triJointName,&boneID) )
    {
        fprintf(stderr,"Could not resolve TRI joint %s \n",triJointName);
        return 0;
    }

    for (BVHJointID j=0; j<modelOriginal->header.numberOfBones; j++)
    {
       if ( modelOriginal->bones[j].info->boneParent == boneID)
       {
          boneChildID = j;
          break;
       }
    }

    if (childOfChild)
    {
      for (BVHJointID j=0; j<modelOriginal->header.numberOfBones; j++)
      {
       if ( modelOriginal->bones[j].info->boneParent == boneChildID)
       {
          boneChildID = j;
          break;
       }
      }
    }


    fprintf(stderr," TRI joint %s ( %u ) -> child %u   \n",triJointName,boneID,boneChildID);

    unsigned int numberOfBones = modelOriginal->header.numberOfBones;
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



    struct Matrix4x4OfFloats dynamicRotation;
    struct TRI_Model   mT = {0};
    struct TRI_Model * mI = modelOriginal;

    if (
         (mI->header.numberOfBones <= boneID ) ||
         (mI->header.numberOfBones <= boneChildID )
       )
    {
         fprintf(stderr,"Bones out of bounds?\n");
         return 0;
    }


    if (
         (mI->bones[boneID].info==0) ||
         (mI->bones[boneChildID].info==0)
       )
    {
         fprintf(stderr,"No bone infos ?\n");
         return 0;
    }

       // valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --track-origins=yes --num-callers=20 --track-fds=yes ./gl3MeshTransform 2>error.txt
       unsigned int testID = 0;
       doModelTransform(&mT,modelOriginal,transformations4x4,numberOfBones * 16 * sizeof(float),1,1,1,0);
       triResult[testID].dX = (float) mI->bones[boneID].info->x - mI->bones[boneChildID].info->x;
       triResult[testID].dY = (float) mI->bones[boneID].info->y - mI->bones[boneChildID].info->y;
       triResult[testID].dZ = (float) mI->bones[boneID].info->z - mI->bones[boneChildID].info->z;



       //-------------------------------------------------------------------------------------------------------------
       ++testID;
       create4x4FMatrixFromEulerAnglesWithRotationOrder(&dynamicRotation, -90.0  , 0.0 , 0.0  , ROTATION_ORDER_ZXY );
       copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
       doModelTransform(&mT,modelOriginal,transformations4x4,numberOfBones * 16 * sizeof(float),1,1,1,0);
       triResult[testID].dX = (float) mI->bones[boneID].info->x - mI->bones[boneChildID].info->x;
       triResult[testID].dY = (float) mI->bones[boneID].info->y - mI->bones[boneChildID].info->y;
       triResult[testID].dZ = (float) mI->bones[boneID].info->z - mI->bones[boneChildID].info->z;
       triResult[testID].value = -90;
       triResult[testID].mID   = boneID;
       //-------------------------------------------------------------------------------------------------------------
       ++testID;
       create4x4FMatrixFromEulerAnglesWithRotationOrder(&dynamicRotation,  90.0  , 0.0 , 0.0  , ROTATION_ORDER_ZXY );
       copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
       doModelTransform(&mT,modelOriginal,transformations4x4,numberOfBones * 16 * sizeof(float),1,1,1,0);
       triResult[testID].dX = (float) mI->bones[boneID].info->x - mI->bones[boneChildID].info->x;
       triResult[testID].dY = (float) mI->bones[boneID].info->y - mI->bones[boneChildID].info->y;
       triResult[testID].dZ = (float) mI->bones[boneID].info->z - mI->bones[boneChildID].info->z;
       triResult[testID].value = 90;
       triResult[testID].mID   = boneID;
       //-------------------------------------------------------------------------------------------------------------
       create4x4FMatrixFromEulerAnglesWithRotationOrder(&dynamicRotation,  0.0  , 0.0 , 0.0  , ROTATION_ORDER_ZXY );
       copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
       //-------------------------------------------------------------------------------------------------------------


       //-------------------------------------------------------------------------------------------------------------
       ++testID;
       create4x4FMatrixFromEulerAnglesWithRotationOrder(&dynamicRotation,  0.0  , -90.0 , 0.0  , ROTATION_ORDER_ZXY );
       copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
       doModelTransform(&mT,modelOriginal,transformations4x4,numberOfBones * 16 * sizeof(float),1,1,1,0);
       triResult[testID].dX = (float) mI->bones[boneID].info->x - mI->bones[boneChildID].info->x;
       triResult[testID].dY = (float) mI->bones[boneID].info->y - mI->bones[boneChildID].info->y;
       triResult[testID].dZ = (float) mI->bones[boneID].info->z - mI->bones[boneChildID].info->z;
       triResult[testID].value = -90;
       triResult[testID].mID   = boneID+1;
       //-------------------------------------------------------------------------------------------------------------
       ++testID;
       create4x4FMatrixFromEulerAnglesWithRotationOrder(&dynamicRotation,  0.0  ,  90.0 , 0.0  , ROTATION_ORDER_ZXY );
       copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
       doModelTransform(&mT,modelOriginal,transformations4x4,numberOfBones * 16 * sizeof(float),1,1,1,0);
       triResult[testID].dX = (float) mI->bones[boneID].info->x - mI->bones[boneChildID].info->x;
       triResult[testID].dY = (float) mI->bones[boneID].info->y - mI->bones[boneChildID].info->y;
       triResult[testID].dZ = (float) mI->bones[boneID].info->z - mI->bones[boneChildID].info->z;
       triResult[testID].value = 90;
       triResult[testID].mID   = boneID+1;
       //-------------------------------------------------------------------------------------------------------------
       create4x4FMatrixFromEulerAnglesWithRotationOrder(&dynamicRotation,  0.0  , 0.0 , 0.0  , ROTATION_ORDER_ZXY );
       copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
       //-------------------------------------------------------------------------------------------------------------


       //-------------------------------------------------------------------------------------------------------------
       ++testID;
       create4x4FMatrixFromEulerAnglesWithRotationOrder(&dynamicRotation,  0.0  ,  0.0 , -90.0  , ROTATION_ORDER_ZXY );
       copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
       doModelTransform(&mT,modelOriginal,transformations4x4,numberOfBones * 16 * sizeof(float),1,1,1,0);
       triResult[testID].dX = (float) mI->bones[boneID].info->x - mI->bones[boneChildID].info->x;
       triResult[testID].dY = (float) mI->bones[boneID].info->y - mI->bones[boneChildID].info->y;
       triResult[testID].dZ = (float) mI->bones[boneID].info->z - mI->bones[boneChildID].info->z;
       triResult[testID].value = -90;
       triResult[testID].mID   = boneID+2;
       //-------------------------------------------------------------------------------------------------------------
       ++testID;
       create4x4FMatrixFromEulerAnglesWithRotationOrder(&dynamicRotation,  0.0  ,  0.0 , 90.0  , ROTATION_ORDER_ZXY );
       copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
       doModelTransform(&mT,modelOriginal,transformations4x4,numberOfBones * 16 * sizeof(float),1,1,1,0);
       triResult[testID].dX = (float) mI->bones[boneID].info->x - mI->bones[boneChildID].info->x;
       triResult[testID].dY = (float) mI->bones[boneID].info->y - mI->bones[boneChildID].info->y;
       triResult[testID].dZ = (float) mI->bones[boneID].info->z - mI->bones[boneChildID].info->z;
       triResult[testID].value = 90;
       triResult[testID].mID   = boneID+2;
       //-------------------------------------------------------------------------------------------------------------
       create4x4FMatrixFromEulerAnglesWithRotationOrder(&dynamicRotation,  0.0  , 0.0 , 0.0  , ROTATION_ORDER_ZXY );
       copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
       //-------------------------------------------------------------------------------------------------------------


         for (int i=0; i<7; i++)
         {
            fprintf(stderr,"Test %u (mID=%u set to %0.2f) ",i,triResult[i].mID,triResult[i].value);
            if (triResult[i].value>0.0) { fprintf(stderr," "); }
            fprintf(stderr," | ");

            //--------------------------------------------------------------
            if (triResult[i].dX>0.0)  {  fprintf(stderr,"+ ");   triResult[i].dX= 1.0; } else
            if (triResult[i].dX<0.0)  {  fprintf(stderr,"- ");   triResult[i].dX=-1.0; } else
                                      {  fprintf(stderr,"0 ");   triResult[i].dX= 0.0; }
            //--------------------------------------------------------------
            if (triResult[i].dY>0.0)  {  fprintf(stderr,"+ ");   triResult[i].dY= 1.0; } else
            if (triResult[i].dY<0.0)  {  fprintf(stderr,"- ");   triResult[i].dY=-1.0; } else
                                      {  fprintf(stderr,"0 ");   triResult[i].dY= 0.0; }
            //--------------------------------------------------------------
            if (triResult[i].dZ>0.0)  {  fprintf(stderr,"+ ");   triResult[i].dZ= 1.0; } else
            if (triResult[i].dZ<0.0)  {  fprintf(stderr,"- ");   triResult[i].dZ=-1.0; } else
                                      {  fprintf(stderr,"0 ");   triResult[i].dZ= 0.0; }
            //--------------------------------------------------------------
            fprintf(stderr,"\n");
         }


      tri_deallocModelInternals(&mT);
      return 1;
}



const static int testsMatch(struct testResult * testA,struct testResult * testB)
{
    if  (
           (testA->dX == testB->dX) &&
           (testA->dY == testB->dY) &&
           (testA->dZ == testB->dZ)
        )
        {
            return 1;
        }
   return 0;
}



const static int alignRotationOfTRIVsBVH(
                                          struct TRI_Model * modelOriginal,
                                          struct BVH_MotionCapture * bvh,
                                          const char * triJointName,
                                          const char * bvhJointName,
                                          int childOfTriChild
                                        )
{
    int matched = 0;
    struct testResult bvhResult[7]={0};
    checkBVHRotation(
                      &bvhResult,
                       bvh,
                       bvhJointName
                    );

    struct testResult triResult[7]={0};
    checkTRIRotation(
                      &triResult,
                      modelOriginal,
                      triJointName,
                      childOfTriChild
                    );

    fprintf(stderr,GREEN "BVH(%s) to TRI(%s)\n" NORMAL,triJointName,bvhJointName);
    if  (testsMatch(&bvhResult[0],&triResult[0]))                                                  { fprintf(stderr,"Neutral Match\n");   }

    // -- -- --  Z axis check  -- -- --
    if  ( (testsMatch(&bvhResult[1],&triResult[1])) && (testsMatch(&bvhResult[2],&triResult[2])) ) { fprintf(stderr,"Match Z ->  Z\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[1],&triResult[2])) && (testsMatch(&bvhResult[2],&triResult[1])) ) { fprintf(stderr,"Match Z -> -Z\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[1],&triResult[3])) && (testsMatch(&bvhResult[2],&triResult[4])) ) { fprintf(stderr,"Match Z ->  X\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[1],&triResult[4])) && (testsMatch(&bvhResult[2],&triResult[3])) ) { fprintf(stderr,"Match Z -> -X\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[1],&triResult[5])) && (testsMatch(&bvhResult[2],&triResult[6])) ) { fprintf(stderr,"Match Z ->  Y\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[1],&triResult[6])) && (testsMatch(&bvhResult[2],&triResult[5])) ) { fprintf(stderr,"Match Z -> -Y\n");  ++matched; }

    // -- -- --  X axis check  -- -- --
    if  ( (testsMatch(&bvhResult[3],&triResult[1])) && (testsMatch(&bvhResult[4],&triResult[2])) ) { fprintf(stderr,"Match X ->  Z\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[3],&triResult[2])) && (testsMatch(&bvhResult[4],&triResult[1])) ) { fprintf(stderr,"Match X -> -Z\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[3],&triResult[3])) && (testsMatch(&bvhResult[4],&triResult[4])) ) { fprintf(stderr,"Match X ->  X\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[3],&triResult[4])) && (testsMatch(&bvhResult[4],&triResult[3])) ) { fprintf(stderr,"Match X -> -X\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[3],&triResult[5])) && (testsMatch(&bvhResult[4],&triResult[6])) ) { fprintf(stderr,"Match X ->  Y\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[3],&triResult[6])) && (testsMatch(&bvhResult[4],&triResult[5])) ) { fprintf(stderr,"Match X -> -Y\n");  ++matched; }

    // -- -- --  Y axis check  -- -- --
    if  ( (testsMatch(&bvhResult[5],&triResult[1])) && (testsMatch(&bvhResult[6],&triResult[2])) ) { fprintf(stderr,"Match Y ->  Z\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[5],&triResult[2])) && (testsMatch(&bvhResult[6],&triResult[1])) ) { fprintf(stderr,"Match Y -> -Z\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[5],&triResult[3])) && (testsMatch(&bvhResult[6],&triResult[4])) ) { fprintf(stderr,"Match Y ->  X\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[5],&triResult[4])) && (testsMatch(&bvhResult[6],&triResult[3])) ) { fprintf(stderr,"Match Y -> -X\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[5],&triResult[5])) && (testsMatch(&bvhResult[6],&triResult[6])) ) { fprintf(stderr,"Match Y ->  Y\n");  ++matched; } else
    if  ( (testsMatch(&bvhResult[5],&triResult[6])) && (testsMatch(&bvhResult[6],&triResult[5])) ) { fprintf(stderr,"Match Y -> -Y\n");  ++matched; }

    if (matched<3) { fprintf(stderr,RED "failed to map..\n" NORMAL); }
    return (matched==3);
}



const static int animateTRIModelUsingBVHArmature(
                                                 struct TRI_Model * modelOutput,
                                                 struct TRI_Model * modelOriginal,
                                                 struct BVH_MotionCapture * bvh,
                                                 unsigned int * lookupTableFromTRIToBVH,
                                                 unsigned int frameID,
                                                 int performTransformsInCPU,
                                                 int printDebugMessages
                                                )
{
    if (modelOriginal==0)  { return 0; }
    if (modelOutput==0)    { return 0; }
    if (bvh==0)            { return 0; }
    //----------------------------------
   if (performTransformsInCPU)
     {
        //If we are doing CPU bone transforms we have to copy our input to our output
        //so that we keep our original model intact
        tri_copyModel(modelOutput, modelOriginal, 1 /*We also want bone data*/,0);
     }

    unsigned int numberOfBones = modelOriginal->header.numberOfBones;

    //----------------------------------------------------

    struct MotionBuffer * frameMotionBuffer = mallocNewMotionBuffer(bvh);

    if (frameMotionBuffer!=0)
       {
        if  ( bvh_copyMotionFrameToMotionBuffer(bvh,frameMotionBuffer,frameID)  )
           {

         /*
        fprintf(stderr,"motionBuffer=");
        for (int i=0; i<frameMotionBuffer->bufferSize; i++)
        {
            fprintf(stderr,"%0.2f ",frameMotionBuffer->motion[i]);
        }
        fprintf(stderr,"\n");*/

        //-------------------------------------------------------------------------------------

        unsigned int transformations4x4Size = numberOfBones * 16;
        float * transformations4x4 = (float *) malloc(sizeof(float) * transformations4x4Size);
        if (transformations4x4==0)
        {
            fprintf(stderr,"Failed to allocate enough memory for bones.. \n");
            freeMotionBuffer(&frameMotionBuffer);
            return 0;
        }

        //Cleanup 4x4 matrix transformation..
        for (unsigned int mID=0; mID<numberOfBones; mID++)
        {
            create4x4FIdentityMatrixDirect(&transformations4x4[mID*16]);
        }


        if (lookupTableFromTRIToBVH!=0)
        {
            {
                float data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER]={0};
                for (unsigned int boneID=0; boneID<numberOfBones; boneID++)
                {
                    if (lookupTableFromTRIToBVH[boneID]!=0)
                    {
                        BVHJointID jID = lookupTableFromTRIToBVH[boneID];

                         //To Setup the dynamic transformation we must first get values from our bvhMotion structure
                         if (bhv_retrieveDataFromMotionBuffer(bvh,jID,frameMotionBuffer->motion,data,sizeof(data)))
                               {
                                 //-----------------------------------------------
                                 //See https://github.com/makehumancommunity/makehuman/blob/master/makehuman/shared/bvh.py#L369
                                 //https://github.com/makehumancommunity/makehuman/blob/master/makehuman/shared/skeleton.py#L1395
                                 //https://github.com/makehumancommunity/makehuman/blob/master/makehuman/core/transformations.py#L317
                                 //svn.blender.org/svnroot/bf-blender/branches/blender-2.47/source/blender/blenkernel/intern/armature.c
                                 //https://developer.blender.org/T39470
                                 //https://github.com/makehumancommunity/makehuman/blob/master/makehuman/shared/skeleton.py#L959

                                 //struct Matrix4x4OfFloats localBone;
                                 //copy4x4FMatrix(localBone.m,modelOriginal->bones[boneID].info->localTransformation);
                                 //struct Matrix4x4OfFloats localBoneInverted;
                                 //invert4x4FMatrix(&localBoneInverted,&localBone);

                                 /*
                                    _Identity = np.identity(4, float)
                                    _RotX = tm.rotation_matrix(math.pi/2, (1,0,0))
                                    _RotY = tm.rotation_matrix(math.pi/2, (0,1,0))
                                    _RotNegX = tm.rotation_matrix(-math.pi/2, (1,0,0))
                                    _RotZ = tm.rotation_matrix(math.pi/2, (0,0,1))
                                    _RotZUpFaceX = np.dot(_RotZ, _RotX)
                                    _RotXY = np.dot(_RotNegX, _RotY)*/

                                 #define M_PI 3.14159265358979323846
                                 /*
                                 fprintf(stderr,"jID %u rX %f rY %f rZ %f \n",jID,
                                         data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X],
                                         data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y],
                                         data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]);*/

                                 float rSignX = -1.0;
                                 float rSignY = -1.0;
                                 float rSignZ = -1.0;
                                 int rotationOrder = ROTATION_ORDER_ZXY;
                                 //  ./gl3MeshTransform --set relbow z 90 --set hip y 180 --set lelbow x 90 --set rknee z -45 --set lshoulder y 45
                                 //  ./gl3MeshTransform --bvhaxis --set relbow z 90 --set hip y 180 --set lelbow x 90 --set rknee z -45 --set lshoulder y 45
                                 //fprintf(stderr,"Rot %u ",rotationOrder);

                                 if (
                                      (strcmp("rshoulder",modelOriginal->bones[boneID].boneName)==0) ||
                                      (strcmp("lshoulder",modelOriginal->bones[boneID].boneName)==0)
                                    )
                                    {
                                      // Z X Y => Y X Z
                                      //Lshoulder TRI Y = BVH -Z
                                      //Lshoulder TRI X = BVH X  ?
                                      //LShoulder TRI Z = BVH -Y
                                      rotationOrder = ROTATION_ORDER_YXZ;
                                      rSignX = -1.0;
                                      rSignY = 1.0;
                                      rSignZ = -1.0;
                                    } else
                                 if (
                                      (strcmp("relbow",modelOriginal->bones[boneID].boneName)==0) ||
                                      (strcmp("lelbow",modelOriginal->bones[boneID].boneName)==0)
                                    )
                                    {
                                      // Z X Y => X -Z -Y
                                      rotationOrder = ROTATION_ORDER_XZY;
                                      rSignX = -1.0;
                                      rSignY = 1.0;
                                      rSignZ = 1.0;
                                    }

                                 if (bvh->jointHierarchy[jID].hasPositionalChannels)
                                   {
                                    //This is one of the new joints with positional channels..
                                    struct Matrix4x4OfFloats mergedTranslation;
                                    create4x4FTranslationMatrix(
                                                                &mergedTranslation,
                                                                (data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_X] - bvh->jointHierarchy[jID].staticTransformation.m[3]  ) / 1,
                                                                (data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Y] - bvh->jointHierarchy[jID].staticTransformation.m[7]  ) / 1,
                                                                (data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Z] - bvh->jointHierarchy[jID].staticTransformation.m[11] ) / 1
                                                               );

                                    struct Matrix4x4OfFloats dynamicRotation;
                                    create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                                                      &dynamicRotation,
                                                                                      rSignX * data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X],
                                                                                      rSignY * data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y],
                                                                                      rSignZ * data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z],
                                                                                      rotationOrder
                                                                                    );

                                    multiplyTwo4x4FMatrices_Naive(
                                                                  &transformations4x4[boneID*16],
                                                                  dynamicRotation.m,
                                                                  mergedTranslation.m
                                                                 );
                                   } else
                                   {
                                    struct Matrix4x4OfFloats dynamicRotation;
                                    create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                                                      &dynamicRotation,
                                                                                      rSignX * data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X],
                                                                                      rSignY * data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y],
                                                                                      rSignZ * data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z],
                                                                                      rotationOrder
                                                                                    );
                                    copy4x4FMatrix(&transformations4x4[boneID*16],dynamicRotation.m);
                                   }
                               } else // Retrieved rotation data ..
                               {
                                 fprintf(stderr,RED "Error: bhv_retrieveDataFromMotionBuffer \n" NORMAL);
                               }
                    }
                }
            } //have resolved joints

           // free(lookupTableFromTRIToBVH);
        }
        else
        {
            fprintf(stderr,RED "Error: Please call createLookupTableFromTRItoBVH to allocate a lookup table from TRI To BVH\n" NORMAL);
        }


        struct TRI_Model modelTemporary = {0};
        //---------------------------------------------------------------
        doModelTransform(
                         &modelTemporary,
                         modelOriginal,
                         transformations4x4,
                         numberOfBones * 16 * sizeof(float), //Each transform has a 4x4 matrix of floats..!
                         1,//Autodetect default matrices for speedup
                         1,//Direct setting of matrices
                         performTransformsInCPU,//Do Transforms, don't just calculate the matrices
                         0 //Default joint convention
                        );
        //---------------------------------------------------------------
        if (performTransformsInCPU)
          {
            //If we are doing CPU bone transforms we have to flatten
            //Temporary model into our output..
            tri_flattenIndexedModel(modelOutput,&modelTemporary);
          }
        tri_deallocModelInternals(&modelTemporary);
        //---------------------------------------------------------------
        free(transformations4x4);
        freeMotionBuffer(&frameMotionBuffer);

        return 1;
       }
    }
    else
    {
        fprintf(stderr,RED "Error: Failed executing bvh transform\n" NORMAL);
    }
    return 0;
}

#endif //TRI_BVH_CONTROLLER_H_INCLUDED
