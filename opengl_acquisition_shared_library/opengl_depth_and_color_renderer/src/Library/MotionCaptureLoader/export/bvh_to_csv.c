#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bvh_to_csv.h"

#include "../bvh_project.h"

#define CONVERT_EULER_TO_RADIANS M_PI/180.0
#define DUMP_SEPERATED_POS_ROT 0
#define DUMP_3D_POSITIONS 0

unsigned int filteredOutCSVBehindPoses=0;
unsigned int filteredOutCSVOutPoses=0;
unsigned int filteredOutCSVPoses=0;

int fileExists(const char * filename)
{
 FILE *fp = fopen(filename,"r");
 if( fp ) { /* exists */ fclose(fp); return 1; }
 return 0;
}



int dumpBVHToCSVHeader(
                       struct BVH_MotionCapture * mc,
                       const char * filename
                      )
{
   if (!fileExists(filename))
   {
     FILE * fp =fopen(filename,"w");

   if (fp!=0)
   {
     #if DUMP_SEPERATED_POS_ROT
     fprintf(fp,"positionX,positionY,positionZ,roll,pitch,yaw,");
     #endif // DUMP_SEPERATED_POS_ROT

     unsigned int jID=0;
     //2D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
          fprintf(fp,"2DX_%s,2DY_%s,",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName);
         }
       }

     #if DUMP_3D_POSITIONS
     //3D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           fprintf(fp,"3DX_%s,3DY_%s,3DZ_%s,",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName);
         }
       }
     #endif // DUMP_3D_POSITIONS


     //Model Configuration
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           unsigned int channelID=0;
           for (channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
           {
            fprintf(
                    fp,"%s_%s,",
                    mc->jointHierarchy[jID].jointName,
                    channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[channelID]]
                   );
           }
         }
       }


     //Append Frame ID
     fprintf(fp,"frameID\n");

     fclose(fp);
     return 1;
   }
  } else
  {
   fprintf(stderr,"We don't need to regenerate the CSV header, it already exists\n");
   return 1;
  }
 return 0;
}


float eulerAngleToRadiansIfNeeded( float eulerAngle , unsigned int isItNeeded)
{
  if (isItNeeded)
  {
    return eulerAngle*CONVERT_EULER_TO_RADIANS;
  }
 return eulerAngle;
}

int dumpBVHToCSVBody(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform * bvhTransform,
                       struct simpleRenderer * renderer,
                       unsigned int fID,
                       const char * filename,
                       unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                       unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                       unsigned int filterWeirdSkeletons,
                       unsigned int encodeRotationsAsRadians
                      )
{
   unsigned int jID=0;

   //-------------------------------------------------
   if (filterOutSkeletonsWithAnyLimbsBehindTheCamera)
   {
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (bvhTransform->joint[jID].isBehindCamera)
         {
           ++filteredOutCSVPoses;
           ++filteredOutCSVBehindPoses;
           //Just counting to reduce spam..
           return 0;
         }
       }
   }//-----------------------------------------------


   //-------------------------------------------------
   if (filterOutSkeletonsWithAnyLimbsOutOfImage)
   {
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
        float x = bvhTransform->joint[jID].pos2D[0];
        float y = bvhTransform->joint[jID].pos2D[1];

        if (
             (x<0.0) || (y<0.0) || (renderer->width<x) || (renderer->height<y)
           )
        {
           ++filteredOutCSVPoses;
           ++filteredOutCSVOutPoses;
           //Just counting to reduce spam..
           return 0;
        }
       }
   }//-----------------------------------------------



   //-------------------------------------------------
   if (filterWeirdSkeletons)
   { //If all x,y 0 filter out
       unsigned int jointCount=0;
       unsigned int jointsInWeirdPositionCount=0;
       for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
        float x = bvhTransform->joint[jID].pos2D[0];
        float y = bvhTransform->joint[jID].pos2D[1];
        ++jointCount;
        if ( (x<0.5) && (y<0.5) )
        {
           ++jointsInWeirdPositionCount;
        }
       }

        if (jointCount==jointsInWeirdPositionCount)
        {
           ++filteredOutCSVPoses;
           return 0;
        }
   }//-----------------------------------------------

   //-------------------------------------------------
   if (encodeRotationsAsRadians)
   {
    fprintf(stderr,"encodeRotationsAsRadians not implemented, please switch it off\n");
    exit(0);
   }//-----------------------------------------------

   FILE * fp = fopen(filename,"a");
   if (fp!=0)
   {
    #if DUMP_SEPERATED_POS_ROT
    fprintf(fp,"%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,",
                 renderer->cameraOffsetPosition[0]*10,
                 renderer->cameraOffsetPosition[1]*10,
                 renderer->cameraOffsetPosition[2]*10,
                 objectRotationOffset[2],
                 objectRotationOffset[1],
                 objectRotationOffset[0]
                 );
     #endif // DUMP_SEPERATED_POS_ROT

     //2D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
          fprintf(
                  fp,"%0.2f,%0.2f,",
                  bvhTransform->joint[jID].pos2D[0],
                  bvhTransform->joint[jID].pos2D[1]
                 );
         }
       }

     #if DUMP_3D_POSITIONS
     //3D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
          fprintf(
                  fp,"%0.2f,%0.2f,%0.2f,",
                  bvhTransform->joint[jID].pos3D[0],
                  bvhTransform->joint[jID].pos3D[1],
                  bvhTransform->joint[jID].pos3D[2]
                 );
         }
       }
     #endif // DUMP_3D_POSITIONS

     //Joint Configuration
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           unsigned int channelID=0;
           for (channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
           {
             unsigned int channelType =  mc->jointHierarchy[jID].channelType[channelID];
             fprintf(
                     fp,"%0.4f,",
                     bvh_getJointChannelAtFrame(mc,jID,fID,channelType)
                    );
           }
         }
       }

     //Append Frame ID
     fprintf(fp,"%u\n",fID);

     fclose(fp);
     return 1;
   }
 return 0;
}



