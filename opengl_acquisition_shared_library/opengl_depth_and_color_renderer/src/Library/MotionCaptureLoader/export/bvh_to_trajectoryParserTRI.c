#include <stdio.h>
#include <string.h>
#include "bvh_to_trajectoryParserTRI.h"
#include "../calculate/bvh_transform.h"

#include "../../../../../../tools/AmMatrix/quaternions.h"


int get_jAssociationID_From_jID(
                                struct BVH_MotionCapture * mc,
                                struct bvhToTRI * bvhtri,
                                unsigned int jID,
                                unsigned int *jAssociationIDResult
                               )
{
   if (mc == 0) {   fprintf(stderr,"get_jAssociationID_From_jID with incorrect BVH structure..\n"); return 0; }
   if (bvhtri == 0) {   fprintf(stderr,"get_jAssociationID_From_jID with incorrect BVH to TRI structure..\n"); return 0; }
   if (jAssociationIDResult == 0) {   fprintf(stderr,"get_jAssociationID_From_jID needs a pointer for result \n"); return 0; }
   //--   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --

   if (jID>=mc->jointHierarchySize)
   {
       fprintf(stderr,"get_jAssociationID_From_jID from incorrect joint ( %u )\n",jID);
       return 0;
   }

  unsigned int jAssociationID=0;
  for (jAssociationID=0; jAssociationID<bvhtri->numberOfJointAssociations; jAssociationID++)
  {
    if  (  ( bvhtri->jointAssociation[jAssociationID].bvhJointName !=0 )  && ( mc->jointHierarchy[jID].jointName!=0 ) )
    {
      //------------------------------
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
    //------------------------------
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
   if (mc == 0) {   fprintf(stderr,"get_jAssociationID_From_jID with incorrect BVH structure..\n"); return 0; }
   if (bvhtri == 0) {   fprintf(stderr,"get_jAssociationID_From_jID with incorrect BVH to TRI structure..\n"); return 0; }
   if (jIDResult == 0) {   fprintf(stderr,"get_jAssociationID_From_jID needs a pointer for result \n"); return 0; }
   //--   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --   --


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


int getAssociatedPositionsAndRotationsForJointID(
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

  float data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER]={0};
  if (bhv_populatePosXYZRotXYZ(mc,jID,fID,data,sizeof(data)))
         {
           *posX=data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_X];
           *posY=data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Y];
           *posZ=data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Z];

           #define INVERT_ANGLES 0

           #if INVERT_ANGLES
            data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X]=data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X]*-1;
            data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y]=data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y]*-1;
            data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]=data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]*-1;
           #endif // INVERT_ANGLES


          //---------------------------------------------------------------------------------------------------
           *rotX = bvhtri->jointAssociation[jAssociationID].rotationOrder[0].sign * data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X];
           *rotX+= bvhtri->jointAssociation[jAssociationID].offset[0];
          //---------------------------------------------------------------------------------------------------
           *rotY = bvhtri->jointAssociation[jAssociationID].rotationOrder[1].sign * data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y];
           *rotY+= bvhtri->jointAssociation[jAssociationID].offset[1];
          //---------------------------------------------------------------------------------------------------
           *rotZ = bvhtri->jointAssociation[jAssociationID].rotationOrder[2].sign * data[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z];
           *rotZ+= bvhtri->jointAssociation[jAssociationID].offset[2];
          //---------------------------------------------------------------------------------------------------
          return 1;
         }

  return 0;
}


int getAssociatedPositionsAndRotationsForJointAssociation(
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
         //bvh_getJointIDFromJointName(
         bvh_getJointIDFromJointNameNocase(
                                      mc,
                                      bvhtri->jointAssociation[jAssociationID].bvhJointName,
                                      &jID
                                     )
        )
      )
       {
        if (
            getAssociatedPositionsAndRotationsForJointID(
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
            !getAssociatedPositionsAndRotationsForJointID(
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
        //---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---
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
         bvh_freeTransform(&bvhTransform);
        //---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---
        #else
              fprintf(
                      fp,"POSE(human,%u,%s,%0.4f,%0.4f,%0.4f)\n",
                      fID,
                      bvhtri->jointAssociation[jAssociationID].triJointName,
                      rotX, //getAssociatedPositionsAndRotationsForJointID takes care of offsets and signs
                      rotY, //getAssociatedPositionsAndRotationsForJointID takes care of offsets and signs
                      rotZ  //getAssociatedPositionsAndRotationsForJointID takes care of offsets and signs
                     );
         //---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---    ---
        #endif
        }
       } else
       { fprintf(fp,"#BVH joint `%s` has no TRI name associated\n",bvhtri->jointAssociation[jAssociationID].bvhJointName); }
    } else
    { fprintf(fp,"#BVH joint `%s` not used/associated\n",bvhtri->jointAssociation[jAssociationID].bvhJointName); }
  }

 return 1;
}






int dumpBVHToTrajectoryParserTRI(
                                  const char * filename ,
                                  struct BVH_MotionCapture * mc,
                                  struct bvhToTRI * bvhtri ,
                                  unsigned int usePosition,
                                  unsigned int includeSpheres
                                )
{
  unsigned int jID=0;
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

    fprintf(fp,"OBJECT_TYPE(floorType,grid)\n");
    fprintf(fp,"OBJECT(floor,floorType,0,255,0,0 ,0, 1.0,1.0,1.0)\n");

    /*
    if (includeSpheres)
    {
    //Instantiate objects that will draw our skeleton
    //------------------------------------------------
     dumpSphereHeader(mc,fp);
    //------------------------------------------------
    }*/

    //fprintf(fp,"\nOBJECT_TYPE(humanMesh,Models/Ammar.tri,http://ammar.gr/models/Ammar.tri)\n");
    fprintf(fp,"\nOBJECT_TYPE(humanMesh,Models/makehuman.tri,http://ammar.gr/mocapnet/makehuman.tri)\n");
    fprintf(fp,"RIGID_OBJECT(human,humanMesh,255,0,0,0,0,1.0,1.0,1.0)\n\n");


    BVHJointID rootJID=0;
     if ( bvh_getRootJointID(mc,&rootJID) )
      {
       fprintf(fp,"OBJECT_ROTATION_ORDER(human,%s)\n",rotationOrderNames[(unsigned int)mc->jointHierarchy[jID].channelRotationOrder]);
      } else
      {
       fprintf(fp,"#OBJECT_ROTATION_ORDER(human,cannot be set because we can't find root bone)\n");
      }

    //-----------------------------------------------------------------------------------------
    unsigned int jAssociationID=0;
    for (jAssociationID=0; jAssociationID<bvhtri->numberOfJointAssociations; jAssociationID++)
       {
         if (
              //bvh_getJointIDFromJointName(
              bvh_getJointIDFromJointNameNocase(
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

    float yOffset=1700;
    float zOffset=4000;

    float posX,posY,posZ,rotX,rotY,rotZ;

    unsigned int fID=0;
    for (fID=0; fID<mc->numberOfFrames; fID++)
    {
      fprintf(fp,"MOVE(floor,%u,-19.231,%0.2f,2699.735,0.0,0.0,0.0,0.0)\n",fID,yOffset);


      unsigned int producedDynamicRotationForRootJoint = 0;

      if ( ( bvh_getRootJointID(mc,&rootJID) )  && (usePosition) )
      {
        fprintf(fp,"#Root joint euler angle order %s\n",rotationOrderNames[(unsigned int)mc->jointHierarchy[rootJID].channelRotationOrder]);

        unsigned int rootjAssociationID=0;
        if (
            //We need to find the correct jAssociationID for the rootJID
            get_jAssociationID_From_jID(
                                         mc,
                                         bvhtri,
                                         rootJID,
                                         &rootjAssociationID
                                        )
           )
        {
         if (
            //We have the correct rootjAssociationID and rootJID, so we retrieve positions and rotations for frame with id fID
             getAssociatedPositionsAndRotationsForJointID(
                                                          mc,
                                                          bvhtri,
                                                          rootJID,
                                                          rootjAssociationID,
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
                    10*posX*-1,(10*posY*-1)+yOffset,(10*posZ)+zOffset,
                    rotX ,rotY, rotZ
                    );
             producedDynamicRotationForRootJoint=1;
           } else { fprintf(stderr,"getAssociatedRotationsForJointID error for RootJID=%u and Frame=%u\n",rootJID,fID); }
        }  else { fprintf( stderr,"Error : get_jAssociationID_From_jID could not get root association..\n" ); }
      }

      //If we did not produce a root dynamic root joint for any reason we will just emmit a static one..
      if (!producedDynamicRotationForRootJoint)
      {
        fprintf(fp,"MOVE(human,%u,58.44,848.85,2105.70,90.00,0.00,0.00)\n",fID);
      }



      dumpBVHJointToTP(fp, mc, bvhtri, fID);
      fprintf(fp,"\n\n");

    //=================================
    /*
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
     */
    //=================================

    }

    fclose(fp);
   return 1;
  }
 return 0;
}





