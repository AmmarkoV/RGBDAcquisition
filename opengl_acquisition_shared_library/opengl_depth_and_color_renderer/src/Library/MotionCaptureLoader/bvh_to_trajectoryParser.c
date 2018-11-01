#include <stdio.h>
#include <string.h>
#include "bvh_to_trajectoryParser.h"
#include "bvh_transform.h"

#include "../../../../../tools/AmMatrix/quaternions.h"



int dumpBVHJointToTP(
                      FILE*fp ,
                      struct BVH_MotionCapture * mc,
                      struct bvhToTRI * bvhtri,
                      unsigned int fID
                    )
{
  unsigned int jID=0;
  unsigned int jAssociationID=0;


  struct BVH_Transform bvhTransform={0};

  bvh_loadTransformForFrame(
                            mc,
                            fID ,
                            &bvhTransform
                           );

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
        //---------------------------------------------------------------------------------------------------
        float X = (bvhtri->jointAssociation[jAssociationID].rotationOrder[0].sign *
        bvh_getJointChannelAtFrame(
                                   mc,
                                   jID,
                                   fID,
                                   bvhtri->jointAssociation[jAssociationID].rotationOrder[0].rotID
                                  ))
         + bvhtri->jointAssociation[jAssociationID].offset[0];
        //---------------------------------------------------------------------------------------------------
        float Y = (bvhtri->jointAssociation[jAssociationID].rotationOrder[1].sign *
        bvh_getJointChannelAtFrame(
                                   mc,
                                   jID,
                                   fID,
                                   bvhtri->jointAssociation[jAssociationID].rotationOrder[1].rotID
                                  ))
        + bvhtri->jointAssociation[jAssociationID].offset[1];
        //---------------------------------------------------------------------------------------------------
        float Z = (bvhtri->jointAssociation[jAssociationID].rotationOrder[2].sign *
        bvh_getJointChannelAtFrame(
                                   mc,
                                   jID,
                                   fID,
                                   bvhtri->jointAssociation[jAssociationID].rotationOrder[2].rotID
                                  ))
        + bvhtri->jointAssociation[jAssociationID].offset[2];
        //---------------------------------------------------------------------------------------------------

        /*
        if (strcmp("lknee",bvhtri->jointAssociation[jAssociationID].bvhJointName)==0)
        {
          fprintf(stderr,"%s offset(%0.4f,%0.4f,%0.4f)\n", bvhtri->jointAssociation[jAssociationID].triJointName,
                  bvhtri->jointAssociation[jAssociationID].offset[0],
                  bvhtri->jointAssociation[jAssociationID].offset[1],
                  bvhtri->jointAssociation[jAssociationID].offset[2] );

          fprintf(stderr,"%s(%u->%0.4f,%0.4f,%0.4f)\n", bvhtri->jointAssociation[jAssociationID].triJointName, fID, X, Y, Z );
        }
        */


        #define USE4X4MAT 0

        #if USE4X4MAT
        fprintf( fp,"POSE4X4(human,%u,%s,", fID, bvhtri->jointAssociation[jAssociationID].triJointName );
        for (unsigned int i=0; i<16; i++)
        {
           fprintf(fp,"%0.4f,",bvhTransform.joint[jID].dynamicRotation[i]);
        }
        fprintf(fp,"\n");
        #else
        fprintf(
                fp,"POSE(human,%u,%s,%0.4f,%0.4f,%0.4f)\n",
                fID, bvhtri->jointAssociation[jAssociationID].triJointName, X, Y, Z
               );
        #endif


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
                    unsigned int fID
                   )
{
      unsigned int jID=0;
      bvh_loadTransformForFrame(
                                mc,
                                fID ,
                                bvhTransform
                               );

     fprintf(fp,"POS(camera,%u,60.0,60.0,252.0,0.0,0.0,0.0,0.0)\n",fID);
     fprintf(fp,"POS(floor,%u,0.0,0.0,0.0,0.0,0.0,0.0,0.0)\n",fID);
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
                                  unsigned int usePosition,
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
    fprintf(fp,"INTERPOLATE_TIME(0)\n");
    fprintf(fp,"MOVE_VIEW(1)\n");

    fprintf(fp,"OBJECT_TYPE(floorType,grid)\n");
    fprintf(fp,"OBJECT(floor,floorType,0,255,0,0 ,0, 1.0,1.0,1.0)\n");
    if (includeSpheres)
    {
    //Instantiate objects that will draw our skeleton
    //------------------------------------------------
     dumpSphereHeader(mc,fp);
    //------------------------------------------------
    }

    fprintf(fp,"\nOBJECT_TYPE(humanMesh,Models/Ammar.tri,http://ammar.gr/models/Ammar.tri)\n");
    fprintf(fp,"RIGID_OBJECT(human,humanMesh, 255,0,0,0,0 ,10.0,10.0,10.0)\n\n");

    //-----------------------------------------------------------------------------------------
    unsigned int jAssociationID=0;
    for (jAssociationID=0; jAssociationID<bvhtri->numberOfJointAssociations; jAssociationID++)
       {
         if (
              bvh_getJointIDFromJointName(
                                           mc,
                                           bvhtri->jointAssociation[jAssociationID].bvhJointName,
                                           &jID
                                          )
             )
             {
               unsigned int channelRotationOrder = mc->jointHierarchy[jID].channelRotationOrder;
               fprintf(
                       fp,"POSE_ROTATION_ORDER(human,%s,%s)\n",
                       bvhtri->jointAssociation[jAssociationID].triJointName,
                       rotationOrderNames[channelRotationOrder]
                      );
             }
       }
    fprintf(fp,"\n\n");
    //-----------------------------------------------------------------------------------------


    for (fID=0; fID<mc->numberOfFrames; fID++)
    {
      fprintf(fp,"MOVE(floor,%u,-19.231,1784.976,2699.735,0.0,0.0,0.0,0.0)\n",fID);

      float dataPos[3];
      float dataRot[3];

      if ( ( bhv_getRootDynamicPosition(mc,fID,dataPos,sizeof(float)*3) ) && (usePosition) )
      {
        bhv_getRootDynamicRotation(mc,fID,dataRot,sizeof(float)*3);
        double euler[3];
        euler[0]=(double) dataRot[0];
        euler[1]=(double) dataRot[1];
        euler[2]=270+(double) dataRot[2];
        double quaternions[4];

        euler2Quaternions(quaternions,euler,qXqYqZqW);

        fprintf(fp,"MOVE(human,%u,%0.2f,%0.2f,%0.2f,%0.5f,%0.5f,%0.5f,%0.5f)\n",fID,
                10*dataPos[0],10*dataPos[1],10*dataPos[2]+3600,
                quaternions[0],quaternions[1],quaternions[2],quaternions[3]);
      } else
      {
        fprintf(fp,"MOVE(human,%u,-19.231,-54.976,2299.735,0.707107,0.707107,0.000000,0.0)\n",fID);
      }

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
  unsigned int fID=0;
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
    fprintf(fp,"INTERPOLATE_TIME(0)\n");
    fprintf(fp,"MOVE_VIEW(1)\n\n");


    fprintf(fp,"OBJECT_TYPE(floorType,grid)\n");
    fprintf(fp,"OBJECT(floor,floorType,0,255,0,0 ,0, 10.0,10.0,10.0)\n");
    //Instantiate objects that will draw our skeleton
    //------------------------------------------------
      dumpSphereHeader(mc,fp);
    //------------------------------------------------

    for (fID=0; fID<mc->numberOfFrames; fID++)
    {
      dumpSphereBody(
                     mc,
                     &bvhTransform,
                     fp ,
                     fID
                    );
    }

    fclose(fp);
    return 1;
  }
 return 0;
}
