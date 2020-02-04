#include "bvh_rename.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void lowercase(char *a)
{
   if (a==0) { return; } 
   while (*a!=0) { *a = tolower(*a); ++a; }
}


void uppercase(char *a)
{
   if (a==0) { return; } 
   while (*a!=0) { *a = toupper(*a); ++a; }
}

void bvh_setLimbFlags(struct BVH_MotionCapture * bvhMotion)
{
 unsigned int jID;
 for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
    unsigned int pJID=bvhMotion->jointHierarchy[jID].parentJoint;
    char * jPN = bvhMotion->jointHierarchy[pJID].jointName;
    char * jN = bvhMotion->jointHierarchy[jID].jointName;

    //----------------------------------------------------------------
    if (
        (strcmp(jN,"abdomen")==0) || (strcmp(jPN,"abdomen")==0) || (strcmp(jN,"chest")==0) || (strcmp(jPN,"chest")==0) || (strcmp(jN,"neck")==0) || (strcmp(jPN,"neck")==0)
        || (strcmp(jN,"hip")==0) || (strcmp(jPN,"hip")==0)
       )
       {
        bvhMotion->jointHierarchy[jID].isAPartOfTorso=1;
       } else
    //----------------------------------------------------------------
    if (
        (strcmp(jN,"neck")==0) || (strcmp(jPN,"neck")==0) || (strcmp(jN,"nose")==0) || (strcmp(jPN,"nose")==0) || (strcmp(jN,"head")==0) || (strcmp(jPN,"head")==0)
       )
       {
        bvhMotion->jointHierarchy[jID].isAPartOfHead=1;
       } else
    //----------------------------------------------------------------
    if (
        (strcmp(jN,"rshoulder")==0) || (strcmp(jPN,"rshoulder")==0) || (strcmp(jN,"relbow")==0) || (strcmp(jPN,"relbow")==0) || (strcmp(jN,"rhand")==0)
       )
       {
        bvhMotion->jointHierarchy[jID].isAPartOfRightArm=1;
       }
    //----------------------------------------------------------------
    if (
        (strcmp(jN,"lshoulder")==0) || (strcmp(jPN,"lshoulder")==0) || (strcmp(jN,"lelbow")==0) || (strcmp(jPN,"lelbow")==0) || (strcmp(jN,"lhand")==0)
       )
       {
        bvhMotion->jointHierarchy[jID].isAPartOfLeftArm=1;
       }
    //----------------------------------------------------------------
    if (
        (strcmp(jN,"rhip")==0) || (strcmp(jN,"rknee")==0) || (strcmp(jN,"rfoot")==0) || (strcmp(jPN,"rfoot")==0)
        )
       {
        bvhMotion->jointHierarchy[jID].isAPartOfRightFoot=1;
       }
      //----------------------------------------------------------------
      if (
          (strcmp(jN,"lhip")==0) || (strcmp(jN,"lknee")==0) || (strcmp(jN,"lfoot")==0) || (strcmp(jPN,"lfoot")==0)
         )
         {
            bvhMotion->jointHierarchy[jID].isAPartOfLeftFoot=1;
         }
     //----------------------------------------------------------------
   }
 }


void bvh_setTorsoImmunityForJoints(struct BVH_MotionCapture * bvhMotion)
{
 unsigned int jID;
 for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
      char * jN = bvhMotion->jointHierarchy[jID].jointName;
      if (
           (strcmp(jN,"abdomen")==0) ||
           (strcmp(jN,"chest")==0) ||
           (strcmp(jN,"neck")==0) ||
           (strcmp(jN,"lcollar")==0) ||
           (strcmp(jN,"rcollar")==0) ||
           (strcmp(jN,"lshoulder")==0) ||
           (strcmp(jN,"rshoulder")==0) ||
           (strcmp(jN,"rbuttock")==0) ||
           (strcmp(jN,"rhip")==0) ||
           (strcmp(jN,"lbuttock")==0) ||
           (strcmp(jN,"lhip")==0) ||
           (strcmp(jN,"hip")==0)
         )
      {
        bvhMotion->jointHierarchy[jID].isImmuneToTorsoOcclusions=1;
      } else
      {
        bvhMotion->jointHierarchy[jID].isImmuneToTorsoOcclusions=0;
      }
   }
}



/*! djb2
This algorithm (k=33) was first reported by dan bernstein many years ago in comp.lang.c.
another version of this algorithm (now favored by bernstein) uses xor: hash(i) = hash(i - 1) * 33 ^ str[i];
the magic of number 33 (why it works better than many other constants, prime or not) has never been adequately explained.
Needless to say , this is our hash function..!
*/
unsigned long hashFunctionJoints(const char *str)
{
 if (str==0) return 0;
 if (str[0]==0) return 0;

 unsigned long hash = 5381; //<- magic
 int c=1;

 while (c != 0)
        {
            c = *str++;
            hash = ((hash << 5) + hash) + c; /* hash * 33 + c */
        }

 return hash;
}



void bvh_updateJointNameHashes(struct BVH_MotionCapture * bvhMotion)
{
 unsigned int jID=0;

  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
   bvhMotion->jointHierarchy[jID].jointNameHash = hashFunctionJoints(bvhMotion->jointHierarchy[jID].jointName);
   }

}


void bvh_renameJointsForCompatibility(struct BVH_MotionCapture * bvhMotion)
{
  if (bvhMotion==0)  { fprintf(stderr,"Cannot rename joints of NULL bvh file\n"); return ; }  
  if (bvhMotion->jointHierarchy==0)  { fprintf(stderr,"Cannot rename joints of NULL bvh hierarchy \n"); return ; }  
  if (bvhMotion->jointHierarchySize==0)  { fprintf(stderr,"Cannot rename joints of empty bvh hierarchy\n"); return ; }  
    
  unsigned int jID=0;

  for (jID=0; jID<bvhMotion->jointHierarchySize; jID++)
   {
     char * jN = bvhMotion->jointHierarchy[jID].jointName;
     if (jN!=0)
     {
     lowercase(jN);

     //-------------------------------------------------------------------------------------------------
     //------------------------------------------TORSO--------------------------------------------------
     //-------------------------------------------------------------------------------------------------
     if  ( (strcmp(jN,"hip")==0) || (strcmp(jN,"hips")==0) )
           { snprintf(jN,MAX_BVH_JOINT_NAME,"hip"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"abdomen")==0) || (strcmp(jN,"spine")==0) || (strcmp(jN,"torso")==0)  || (strcmp(jN,"waist")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"abdomen"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"chest")==0) || (strcmp(jN,"chest2")==0) || (strcmp(jN,"spine1")==0)  || (strcmp(jN,"spine2")==0) || (strcmp(jN,"torso2")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"chest"); }
     //-------------------------------------------------------------------------------------------------


     //-------------------------------------------------------------------------------------------------
     //---------------------------------------LEFT FOOT------------------------------------------------
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"lefthip")==0) || (strcmp(jN,"leftupleg")==0) || (strcmp(jN,"lthigh")==0)  || (strcmp(jN,"leftupperLeg")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lhip"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftknee")==0) || (strcmp(jN,"leftlowleg")==0) || (strcmp(jN,"leftleg")==0)  || (strcmp(jN,"lshin")==0) || (strcmp(jN,"leftlowerleg")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lknee"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftankle")==0) || (strcmp(jN,"leftfoot")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lfoot"); }
     //-------------------------------------------------------------------------------------------------


     //-------------------------------------------------------------------------------------------------
     //---------------------------------------RIGHT FOOT------------------------------------------------
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"righthip")==0) || (strcmp(jN,"rightupleg")==0) || (strcmp(jN,"rthigh")==0)  || (strcmp(jN,"rightupperleg")==0)
          )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rhip"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightknee")==0) || (strcmp(jN,"rightlowleg")==0) || (strcmp(jN,"rightleg")==0)  || (strcmp(jN,"rshin")==0) || (strcmp(jN,"rightlowerleg")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rknee"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightankle")==0) || (strcmp(jN,"rightfoot")==0) || (strcmp(jN,"rfoot")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rfoot"); }
     //-------------------------------------------------------------------------------------------------


     //-------------------------------------------------------------------------------------------------
     //---------------------------------------LEFT HAND------------------------------------------------
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftcollar")==0) || (strcmp(jN,"lcollar")==0) || (strcmp(jN,"leftshoulder")==0)  || (strcmp(jN,"lcolr")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lcollar"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftuparm")==0) || (strcmp(jN,"leftarm")==0) || (strcmp(jN,"lshldr")==0)  || (strcmp(jN,"leftshoulder")==0) || (strcmp(jN,"leftupperarm")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lshoulder"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftelbow")==0) || (strcmp(jN,"leftlowarm")==0) || (strcmp(jN,"leftforearm")==0)  || (strcmp(jN,"lforearm")==0) || (strcmp(jN,"leftlowerarm")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lelbow"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"leftwrist")==0) || (strcmp(jN,"lefthand")==0) || (strcmp(jN,"lhand")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"lhand"); }


     //-------------------------------------------------------------------------------------------------
     //---------------------------------------RIGHT HAND------------------------------------------------
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightcollar")==0) || (strcmp(jN,"rcollar")==0) || (strcmp(jN,"rightshoulder")==0)  || (strcmp(jN,"rcolr")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rcollar"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightuparm")==0) || (strcmp(jN,"rightarm")==0) || (strcmp(jN,"rshldr")==0)  || (strcmp(jN,"rightshoulder")==0) || (strcmp(jN,"rightupperarm")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rshoulder"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightelbow")==0) || (strcmp(jN,"rightlowarm")==0) || (strcmp(jN,"rightforearm")==0)  || (strcmp(jN,"rforearm")==0) || (strcmp(jN,"rightlowerarm")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"relbow"); }
     //-------------------------------------------------------------------------------------------------
     if  (
           (strcmp(jN,"rightwrist")==0) || (strcmp(jN,"righthand")==0) || (strcmp(jN,"rhand")==0)
         )
            { snprintf(jN,MAX_BVH_JOINT_NAME,"rhand"); }
     //-------------------------------------------------------------------------------------------------
   }
   }


  bvh_setLimbFlags(bvhMotion);
  bvh_setTorsoImmunityForJoints(bvhMotion);
  bvh_updateJointNameHashes(bvhMotion);
}


