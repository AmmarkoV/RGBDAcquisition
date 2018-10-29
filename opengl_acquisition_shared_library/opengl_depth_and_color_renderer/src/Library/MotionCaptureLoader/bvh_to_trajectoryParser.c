#include <stdio.h>
#include <string.h>
#include "bvh_to_trajectoryParser.h"
#include "bvh_transform.h"


const char * bvhName[] =
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

const char * triName[] =
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



int dumpBVHJointToTP(
                      FILE*fp ,
                      struct BVH_MotionCapture * mc,
                      struct bvhToTRI * bvhtri,
                      unsigned int fID
                    )
{
  unsigned int jID=0;
  unsigned int jAssociationID=0;

  for (jAssociationID=0; jAssociationID<bvhtri->numberOfJointAssociations; jAssociationID++)
  {
    if (
        (bvhtri->jointAssociation[jAssociationID].useJoint) &&
        (
         bvh_getJointIDFromJointName(
                                      mc,
                                      bvhtri->jointAssociation[jAssociationID].bvhJointName,
                                      &jID
                                     )
        )
       )
    {
     if (strlen(bvhtri->jointAssociation[jAssociationID].triJointName)>0)
       {
         fprintf(
                 fp,"POSE(human,%u,%s,%0.4f,%0.4f,%0.4f)\n",
                 fID,
                 bvhtri->jointAssociation[jAssociationID].triJointName,
                 -1*bvh_getJointRotationXAtFrame(mc,jID,fID),
                 -1*bvh_getJointRotationYAtFrame(mc,jID,fID),
                 -1*bvh_getJointRotationZAtFrame(mc,jID,fID)
                );
       } else
       { fprintf(fp,"#BVH joint `%s` has no TRI name associated\n",bvhtri->jointAssociation[jAssociationID].bvhJointName); }
    }
   else
    { fprintf(fp,"#BVH joint `%s` not used/associated\n",bvhtri->jointAssociation[jAssociationID].bvhJointName); }
  }

 return 1;
}






void dumpSphereHeader(struct BVH_MotionCapture * mc,FILE *fp)
{
 unsigned int jID;
    for (jID=0; jID<mc->jointHierarchySize; jID++)
    {
      if ( mc->jointHierarchy[jID].isEndSite )  { fprintf(fp,"OBJECT_TYPE(sT%u,cube)\n",jID);   } else
                                                { fprintf(fp,"OBJECT_TYPE(sT%u,sphere)\n",jID); }

      if ( mc->jointHierarchy[jID].isEndSite )  { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 0,255,0,0,0 ,3.0,3.0,3.0)\n",jID,jID);   } else
      if ( mc->jointHierarchy[jID].isRoot )     { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 255,255,0,0,0,4.5,4.5,4.5)\n",jID,jID); } else
                                                { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 255,0,0,0,0 ,2.5,2.5,2.5)\n",jID,jID);   }


      if (bhv_jointHasParent(mc,jID))
      {
        fprintf(fp,"CONNECTOR(s%u,s%u, 255,255,0,100, 3.5)\n",jID,mc->jointHierarchy[jID].parentJoint);
      }
    }
    fprintf(fp,"\n");
}


void dumpSphereBody(
                    struct BVH_MotionCapture * mc,
                    struct BVH_Transform * bvhTransform,
                    FILE *fp ,
                    unsigned int jID,
                    unsigned int fID
                   )
{
      bvh_loadTransformForFrame(
                                mc,
                                fID ,
                                bvhTransform
                               );

     fprintf(fp,"POS(camera,%u,   60.0, 60.0, 252.0 , 0.0, 0.0, 0.0,0.0 )\n",fID);
     fprintf(fp,"POS(floor,%u,00.0,00.0,0.0 , 0.0, 0.0, 0.0,0.0 )\n",fID);
     for (jID=0; jID<mc->jointHierarchySize; jID++)
     {
      fprintf(
              fp,"POS(s%u,%u,%0.4f,%0.4f,%0.4f,0,0,0,0)\n",jID,fID,
              bvhTransform->joint[jID].pos[0],
              bvhTransform->joint[jID].pos[1],
              bvhTransform->joint[jID].pos[2]
             );
     }
     fprintf(fp,"\n");
}



int dumpBVHToTrajectoryParserTRI(
                                  const char * filename ,
                                  struct BVH_MotionCapture * mc,
                                  struct bvhToTRI * bvhtri ,
                                  unsigned int includeSpheres
                                )
{
  struct BVH_Transform bvhTransform={0};
  unsigned int jID=0,fID=0;
  FILE * fp = fopen(filename,"w");

  if (fp!=0)
  {
    fprintf(fp,"#Auto generated using BVHTester to render file : %s to scene : %s\n",mc->fileName,filename);
    fprintf(fp,"#https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer\n");
    fprintf(fp,"BACKGROUND(63,114,182)\n");

    fprintf(fp,"#INCLUDE(Scenes/renderLikeMBVRH.conf)\n");
    fprintf(fp,"#This is the way to render like the mbv renderer :)\n");
    fprintf(fp,"AUTOREFRESH(1500)\n");

    fprintf(fp,"NEAR_CLIP(0.1)\n");
    fprintf(fp,"FAR_CLIP(1000)\n");

    fprintf(fp,"#Bring our world to the MBV coordinate system\n");
    fprintf(fp,"SCALE_WORLD(-0.01,-0.01,0.01)\n");
    fprintf(fp,"MAP_ROTATIONS(-1,-1,1,zxy)\n");
    fprintf(fp,"OFFSET_ROTATIONS(0,0,0)\n");
    fprintf(fp,"EMULATE_PROJECTION_MATRIX(535.423889 , 0.0 , 320.0 , 0.0 , 533.48468, 240.0 , 0 , 1)\n");

    fprintf(fp,"SILENT(1)\n");
    fprintf(fp,"RATE(120)\n");
    fprintf(fp,"INTERPOLATE_TIME(1)\n");
    fprintf(fp,"MOVE_VIEW(1)\n");

    if (includeSpheres)
    {
    fprintf(fp,"OBJECT_TYPE(floorType,grid)\n");
    fprintf(fp,"OBJECT(floor,floorType,0,255,0,0 ,0, 10.0,10.0,10.0)\n");
    //Instantiate objects that will draw our skeleton
    //------------------------------------------------
     dumpSphereHeader(mc,fp);
    //------------------------------------------------
    }

    fprintf(fp,"\nOBJECT_TYPE(humanMesh,Models/AmmarH.tri)\n");
    fprintf(fp,"RIGID_OBJECT(human,humanMesh, 255,0,0,0,0 ,10.0,10.0,10.0)\n\n");

    for (fID=0; fID<mc->numberOfFrames; fID++)
    {
      fprintf(fp,"MOVE(human,%u,-19.231,-54.976,2299.735,0.707107,0.707107,0.000000,0.0)\n",fID);
      dumpBVHJointToTP(fp, mc, bvhtri, fID);
      fprintf(fp,"\n\n");

     if (includeSpheres)
      {
       for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
        dumpSphereBody(
                      mc,
                      &bvhTransform,
                      fp ,
                      jID,
                      fID
                     );
       }
      }
    }

    fclose(fp);
   return 1;
  }
 return 0;
}








int dumpBVHToTrajectoryParser(const char * filename , struct BVH_MotionCapture * mc)
{
  unsigned int jID=0,fID=0;
  FILE * fp = fopen(filename,"w");

  struct BVH_Transform bvhTransform={0};

  if (fp!=0)
  {
    fprintf(fp,"#Auto generated using BVHTester to render file : %s to scene : %s\n",mc->fileName,filename);
    fprintf(fp,"#https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer\n");
    fprintf(fp,"AUTOREFRESH(1500)\n");
    fprintf(fp,"BACKGROUND(63,114,182)\n");


    //fprintf(fp,"SCALE_WORLD(-1,1,1)\n");
    fprintf(fp,"NEAR_CLIP(0.1)\n");
    fprintf(fp,"FAR_CLIP(1000)\n");
    fprintf(fp,"SILENT(1)\n");
    fprintf(fp,"RATE(120)\n");
    fprintf(fp,"INTERPOLATE_TIME(1)\n");
    fprintf(fp,"MOVE_VIEW(1)\n\n");


    fprintf(fp,"OBJECT_TYPE(floorType,grid)\n");
    fprintf(fp,"OBJECT(floor,floorType,0,255,0,0 ,0, 10.0,10.0,10.0)\n");
    //Instantiate objects that will draw our skeleton
    //------------------------------------------------
      dumpSphereHeader(mc,fp);
    //------------------------------------------------

    for (fID=0; fID<mc->numberOfFrames; fID++)
    {
     for (jID=0; jID<mc->jointHierarchySize; jID++)
     {
      dumpSphereBody(
                     mc,
                     &bvhTransform,
                     fp ,
                     jID,
                     fID
                    );
     }
    }

    fclose(fp);
    return 1;
  }
 return 0;
}
