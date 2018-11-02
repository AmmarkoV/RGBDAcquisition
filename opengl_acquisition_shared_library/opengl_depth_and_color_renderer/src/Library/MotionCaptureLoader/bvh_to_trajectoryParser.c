#include <stdio.h>
#include <string.h>
#include "bvh_to_trajectoryParser.h"
#include "bvh_transform.h"

#include "../../../../../tools/AmMatrix/quaternions.h"


int get_jAssociationID_From_jID(
                                struct BVH_MotionCapture * mc,
                                struct bvhToTRI * bvhtri,
                                unsigned int jID,
                                unsigned int *jAssociationIDResult
                               )
{
  unsigned int jAssociationID=0;
  for (jAssociationID=0; jAssociationID<bvhtri->numberOfJointAssociations; jAssociationID++)
  {
    if (
        strcmp(
                 bvhtri->jointAssociation[jAssociationID].bvhJointName ,
                 mc->jointHierarchy[jID].jointName
              ) ==0
        )
    {
      *jAssociationIDResult=jAssociationID;
      return 1;
    }
  }

 return 0;
}


int get_jID_From_jAssociationID(
                                struct BVH_MotionCapture * mc,
                                struct bvhToTRI * bvhtri,
                                unsigned int jAssociationID,
                                unsigned int *jIDResult
                               )
{
  unsigned int jID=0;
  for (jID=0; jID<mc->jointHierarchySize; jID++)
  {
    if (
        strcmp(
                 bvhtri->jointAssociation[jAssociationID].bvhJointName ,
                 mc->jointHierarchy[jID].jointName
              ) ==0
        )
    {
      *jIDResult=jID;
      return 1;
    }
  }

 return 0;
}


int getAssociatedPositionRotationsForJointID(
                                             struct BVH_MotionCapture * mc,
                                             struct bvhToTRI * bvhtri,
                                             unsigned int jID,
                                             unsigned int jAssociationID,
                                             unsigned int fID,
                                             float *posX,
                                             float *posY,
                                             float *posZ,
                                             float *rotX,
                                             float *rotY,
                                             float *rotZ
                                   )
{
  if (strcmp(mc->jointHierarchy[jID].jointName,bvhtri->jointAssociation[jAssociationID].bvhJointName)!=0)
        {
          fprintf(
                  stderr,"getAssociatedPositionRotationsForJointID : Error Root joint association (%s) does not have the same name as root joint (%s)..\n",
                  bvhtri->jointAssociation[jAssociationID].bvhJointName,
                  mc->jointHierarchy[jID].jointName
                 );
          return 0;
        }

  float data[8]={0};
  if (bhv_populatePosXYZRotXYZ(mc,jID,fID,data,sizeof(data)))
         {
           *posX=data[0];
           *posY=data[1];
           *posZ=data[2];

          //---------------------------------------------------------------------------------------------------
           *rotX = bvhtri->jointAssociation[jAssociationID].rotationOrder[0].sign * data[3];
           *rotX+= bvhtri->jointAssociation[jAssociationID].offset[0];
          //---------------------------------------------------------------------------------------------------
           *rotY = bvhtri->jointAssociation[jAssociationID].rotationOrder[1].sign * data[4];
           *rotY+= bvhtri->jointAssociation[jAssociationID].offset[1];
          //---------------------------------------------------------------------------------------------------
           *rotZ = bvhtri->jointAssociation[jAssociationID].rotationOrder[2].sign * data[5];
           *rotZ+= bvhtri->jointAssociation[jAssociationID].offset[2];
          //---------------------------------------------------------------------------------------------------
          return 1;
         }

  return 0;
}


int getAssociatedRotationsForJointAssociation(
                                              struct BVH_MotionCapture * mc,
                                              struct bvhToTRI * bvhtri,
                                              unsigned int jAssociationID,
                                              unsigned int fID,
                                              float *posX,
                                              float *posY,
                                              float *posZ,
                                              float *rotX,
                                              float *rotY,
                                              float *rotZ
                                             )
{
  unsigned int jID=0;
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
        if (
            getAssociatedPositionRotationsForJointID(
                                                     mc,
                                                     bvhtri,
                                                     jID,
                                                     jAssociationID,
                                                     fID,
                                                     posX,
                                                     posY,
                                                     posZ,
                                                     rotX,
                                                     rotY,
                                                     rotZ
                                                    )
            )
         {
           return 1;
         }
         else { fprintf(stderr,"Error getAssociatedRotationsForJointID jID=%u @ fID=%u\n",jID,fID); }
       } else { fprintf(stderr,"Error extracting getting joind id for joint name `%s`\n",bvhtri->jointAssociation[jAssociationID].bvhJointName); }
  return 0;
}


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
         get_jID_From_jAssociationID(
                                     mc,
                                     bvhtri,
                                     jAssociationID,
                                     &jID
                                    )
        )
       )
    {
     if (strlen(bvhtri->jointAssociation[jAssociationID].triJointName)>0)
       {
        float posX,posY,posZ,rotX,rotY,rotZ;
        if (
            !getAssociatedPositionRotationsForJointID(
                                                      mc,
                                                      bvhtri,
                                                      jID,
                                                      jAssociationID,
                                                      fID,
                                                      &posX,
                                                      &posY,
                                                      &posZ,
                                                      &rotX,
                                                      &rotY,
                                                      &rotZ
                                                      )
           )
        {
           fprintf(stderr,"getAssociatedRotationsForJointID error for jID=%u and Frame=%u\n",jID,fID);
        }

        #define USE4X4MAT 0


        if (mc->jointHierarchy[jID].isRoot)
        {
         fprintf(
                 fp,"#ALREADY SET POSE(human,%u,%s,%0.4f,%0.4f,%0.4f)\n",
                 fID,
                 bvhtri->jointAssociation[jAssociationID].triJointName,
                 rotX,rotY,rotZ
                );
        } else
        {
        #if USE4X4MAT
        struct BVH_Transform bvhTransform={0};
        bvh_loadTransformForFrame(
                                  mc,
                                  fID ,
                                  &bvhTransform
                                  );
         fprintf( fp,"POSE4X4(human,%u,%s,", fID, bvhtri->jointAssociation[jAssociationID].triJointName );
         for (unsigned int i=0; i<16; i++)
         {
           fprintf(fp,"%0.4f,",bvhTransform.joint[jID].dynamicRotation[i]);
         }
         fprintf(fp,"\n");
        #else
         fprintf(
                 fp,"POSE(human,%u,%s,%0.4f,%0.4f,%0.4f)\n",
                 fID,
                 bvhtri->jointAssociation[jAssociationID].triJointName,
                 rotX, rotY, rotZ
                );
        #endif
        }
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
    fprintf(fp,"MOVE_ROTATION_ORDER(human,ZYX) TODO\n");

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

    float posX,posY,posZ,rotX,rotY,rotZ;
    BVHJointID rootJID=0;
    for (fID=0; fID<mc->numberOfFrames; fID++)
    {
      fprintf(fp,"MOVE(floor,%u,-19.231,1784.976,2699.735,0.0,0.0,0.0,0.0)\n",fID);


      unsigned int producedDynamicRotationForRootJoint = 0;

      if ( ( bvh_getRootJointID(mc,&rootJID) )  && (usePosition) )
      {
        fprintf(fp,"#Root joint euler angle order %s\n",rotationOrderNames[(unsigned int)mc->jointHierarchy[rootJID].channelRotationOrder]);

        if (
            //We need to find the correct jAssociationID for the rootJID
            get_jAssociationID_From_jID(
                                         mc,
                                         bvhtri,
                                         rootJID,
                                         &jAssociationID
                                        )
           )
        {
         if (
            //We have the correct jAssociationID and rootJID, so we retrieve positions and rotations for frame with id fID
             getAssociatedPositionRotationsForJointID(
                                                      mc,
                                                      bvhtri,
                                                      rootJID,
                                                      jAssociationID,
                                                      fID,
                                                      &posX, &posY, &posZ,
                                                      &rotX, &rotY, &rotZ
                                                      )
           )
           {
            //Great, we have everything so we write it down to our output..
            fprintf(
                    fp,"MOVE(human,%u,%0.2f,%0.2f,%0.2f,%0.5f,%0.5f,%0.5f)\n",
                    fID,
                    10*posX, 10*posY, 10*posZ+3600,
                    rotX ,rotY, rotZ
                    );
             producedDynamicRotationForRootJoint=1;
           } else { fprintf(stderr,"getAssociatedRotationsForJointID error for RootJID=%u and Frame=%u\n",rootJID,fID); }
        }  else { fprintf( stderr,"Error : get_jAssociationID_From_jID could not get root association..\n" ); }
      }

      //If we did not produce a root dynamic root joint for any reason we will just emmit a static one..
      if (!producedDynamicRotationForRootJoint)
      {
        fprintf(fp,"MOVE(human,%u,-19.231,-54.976,2299.735,0.707107,0.707107,0.0,0.0)\n",fID);
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
