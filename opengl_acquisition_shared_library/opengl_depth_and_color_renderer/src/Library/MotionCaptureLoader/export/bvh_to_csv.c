#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bvh_to_csv.h"

#include "../bvh_project.h"

#define CONVERT_EULER_TO_RADIANS M_PI/180.0
#define DUMP_3D_POSITIONS 0

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
     fprintf(fp,"positionX,positionY,positionZ,roll,pitch,yaw,");


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


     //3D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           fprintf(fp,"%s_%s,%s_%s,%s_%s,",
                   channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[0]],
                   mc->jointHierarchy[jID].jointName,
                   channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[1]],
                   mc->jointHierarchy[jID].jointName,
                   channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[2]],
                   mc->jointHierarchy[jID].jointName);
         }
       }
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
                       float * objectRotationOffset,
                       unsigned int fID,
                       const char * filename,
                       unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
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
           //Just counting to reduce spam..
           /*
           fprintf(
                   stderr,"Filtering out %0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f, joint %s is behind camera\n",
                   renderer->cameraOffsetPosition[0]*10,
                   renderer->cameraOffsetPosition[1]*10,
                   renderer->cameraOffsetPosition[2]*10,
                   objectRotationOffset[2],
                   objectRotationOffset[1],
                   objectRotationOffset[0],
                   mc->jointHierarchy[jID].jointName
                  );
           */
           return 0;
         }
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
    fprintf(fp,"%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,",
                 renderer->cameraOffsetPosition[0]*10,
                 renderer->cameraOffsetPosition[1]*10,
                 renderer->cameraOffsetPosition[2]*10,
                 objectRotationOffset[2],
                 objectRotationOffset[1],
                 objectRotationOffset[0]
                 );


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
           fprintf(
                   fp,"%0.2f,%0.2f,%0.2f,",
                   bvh_getJointChannelAtFrame(mc,jID,fID,(unsigned int) mc->jointHierarchy[jID].channelType[0]),
                   bvh_getJointChannelAtFrame(mc,jID,fID,(unsigned int) mc->jointHierarchy[jID].channelType[1]),
                   bvh_getJointChannelAtFrame(mc,jID,fID,(unsigned int) mc->jointHierarchy[jID].channelType[2])
                   );
         }
       }

     fprintf(fp,"%u\n",fID);

     fclose(fp);
     return 1;
   }
 return 0;
}



