#include <stdio.h>
#include <string.h>
#include "bvh_to_trajectoryParser.h"
#include "bvh_transform.h"


static const char * bvhName[] =
{
    "Hips",
    "Chest",
    "Neck",
    "Head",
    "LeftCollar",
    "LeftUpArm",
    "LeftLowArm",
    "LeftHand",
    "RightCollar",
    "RightUpArm",
    "RightLowArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLowLeg",
    "LeftFoot",
    "RightUpLeg",
    "RightLowLeg",
    "RightFoot",
//=================
    "Unknown"
};

static const char * triName[] =
{
    "JtSpineA",
    "Chest",
    "JtNeckB",
    "Head",
    "LeftCollar",
    "JtShoulderLf",
    "JtElbowLf",
    "LeftHand",
    "RightCollar",
    "RightUpArm",
    "JtShoulderRt",
    "RightHand",
    "JtHipLf",
    "JtKneeLf",
    "LeftFoot",
    "JtHipRt",
    "JtKneeRt",
    "RightFoot",
//=================
    "Unknown"
};



int dumpBVHJointToTP(FILE*fp , struct BVH_MotionCapture * mc , unsigned int fID, unsigned int jID)
{
  const char * label=0;
  unsigned int jName=0;

   for (jName=0; jName<17; jName++)
    {
      if ( strcmp(mc->jointHierarchy[jID].jointName,bvhName[jName]) == 0 )
      {
        label=triName[jName];
        break;
      }
    }

  if (label==0) { return 0; }


  fprintf(
          fp,"POSE(human,%u,%s,%0.4f,%0.4f,%0.4f)\n",
          fID,
          label,
          bvh_getJointRotationXAtFrame(mc,jID,fID),
          bvh_getJointRotationYAtFrame(mc,jID,fID),
          bvh_getJointRotationZAtFrame(mc,jID,fID)
         );
}


int dumpBVHToTrajectoryParserTRI(const char * filename , struct BVH_MotionCapture * mc)
{
  unsigned int jID=0,fID;
  FILE * fp = fopen(filename,"w");

  if (fp!=0)
  {
    fprintf(fp,"#INCLUDE(Scenes/renderLikeMBVRH.conf)\n");
    fprintf(fp,"#This is the way to render like the mbv renderer :)\n");
    fprintf(fp,"AUTOREFRESH(1500)\n");
    fprintf(fp,"BACKGROUND(0,0,0)\n");

    fprintf(fp,"#Bring our world to the MBV coordinate system\n");
    fprintf(fp,"SCALE_WORLD(-0.01,-0.01,0.01)\n");
    fprintf(fp,"MAP_ROTATIONS(-1,-1,1,zxy)\n");
    fprintf(fp,"OFFSET_ROTATIONS(0,0,0)\n");
    fprintf(fp,"EMULATE_PROJECTION_MATRIX(535.423889 , 0.0 , 320.0 , 0.0 , 533.48468, 240.0 , 0 , 1)\n");

    fprintf(fp,"SILENT(1)\n");
    fprintf(fp,"RATE(100)\n");
    fprintf(fp,"INTERPOLATE_TIME(1)\n");
    fprintf(fp,"MOVE_VIEW(1)\n");

    fprintf(fp,"\nOBJECT_TYPE(humanMesh,Models/AmmarH.tri)\n");
    fprintf(fp,"RIGID_OBJECT(human,humanMesh, 255,0,0,0,0 ,10.0,10.0,10.0)\n\n");

    for (fID=0; fID<mc->numberOfFrames; fID++)
    {
      fprintf(fp,"MOVE(human,%u,-19.231,-54.976,2299.735,0.707107,0.707107,0.000000,0.0)\n",fID);
      for (jID=0; jID<mc->jointHierarchySize; jID++)
      {
        dumpBVHJointToTP(fp , mc , fID, jID);
      }
      fprintf(fp,"\n");
    }




/*
fprintf(fp,"POSE(human,0,JtKneeRt,-5.591,0.655,0.000)\n");
POSE(human,0,JtShoulderLf,0.275,0.867,18.693)
POSE(human,0,JtShoulderRt,0.520,0.892,8.911)
POSE(human,0,JtElbowLf,-0.637,0.064,0.000)
POSE(human,0,JtElbowRt,0.196,0.915,0.000)
POSE(human,0,JtHipLf,-0.000,0.000,0.000)
POSE(human,0,JtHipRt,0.000,0.000,0.000)
POSE(human,0,JtNeckB,-0.901,-0.605,0.286)
POSE(human,0,JtSpineA,0.001,0.000,0.000)
*/


      fclose(fp);
      return 1;
  }
 return 0;
}















int dumpBVHToTrajectoryParser(const char * filename , struct BVH_MotionCapture * mc)
{
  unsigned int jID=0,fID=0;
  FILE * fp = fopen(filename,"w");

  struct BVH_Transform bvhTransform;

  if (fp!=0)
  {
    fprintf(fp,"#This is the way to render like the mbv renderer :)\n");
    fprintf(fp,"AUTOREFRESH(1500)\n");
    fprintf(fp,"BACKGROUND(0,0,0)\n");

    fprintf(fp,"SILENT(1)\n");
    fprintf(fp,"RATE(100)\n");
    fprintf(fp,"INTERPOLATE_TIME(1)\n");
    fprintf(fp,"MOVE_VIEW(1)\n");

    for (jID=0; jID<mc->jointHierarchySize; jID++)
    {
      if ( mc->jointHierarchy[jID].isEndSite )  { fprintf(fp,"OBJECT_TYPE(sT%u,cube)\n",jID);   } else
                                                { fprintf(fp,"OBJECT_TYPE(sT%u,sphere)\n",jID); }

      if ( mc->jointHierarchy[jID].isEndSite )  { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 0,255,0,0,0 ,0.5,0.5,0.5)\n",jID,jID);    } else
      if ( mc->jointHierarchy[jID].isRoot )     { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 255,255,0,0,0 ,0.5,0.5,0.5)\n",jID,jID); } else
                                                { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 255,0,0,0,0 ,0.5,0.5,0.5)\n",jID,jID); }


      if (bhv_jointHasParent(mc,jID))
      {
        fprintf(fp,"CONNECTOR(s%u,s%u, 255,255,0, 1.0)\n",jID,mc->jointHierarchy[jID].parentJoint);
      }
    }
    fprintf(fp,"\n");

    for (fID=0; fID<mc->numberOfFrames; fID++)
    {
      bvh_loadTransformForFrame(
                                mc,
                                fID ,
                                &bvhTransform
                               );

     fprintf(fp,"POS(camera,%u,   0.0, 40.0, 122.0 , 0.0, 0.0,0.0,0.0 )\n",fID);
     for (jID=0; jID<mc->jointHierarchySize; jID++)
     {
     /*
      fprintf(
              fp,"POS(s%u,%u,%0.4f,%0.4f,%0.4f,0,0,0,0)\n",jID,fID,
              mc->jointHierarchy[jID].offset[0],
              mc->jointHierarchy[jID].offset[1],
              mc->jointHierarchy[jID].offset[2]
             );*/

      fprintf(
              fp,"POS(s%u,%u,%0.4f,%0.4f,%0.4f,0,0,0,0)\n",jID,fID,
              bvhTransform.joint[jID].pos[0],
              bvhTransform.joint[jID].pos[1],
              bvhTransform.joint[jID].pos[2]
             );
     }
     fprintf(fp,"\n");
    }



    //OBJECT(left_sph0,obj_type,0,255,0,0 ,0, 0.14,0.14,0.14)

    fclose(fp);
    return 1;
  }
 return 0;
}
