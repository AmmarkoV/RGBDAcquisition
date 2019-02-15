#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bvh_to_csv.h"

#include "../bvh_project.h"

#define CONVERT_EULER_TO_RADIANS M_PI/180.0
#define DUMP_SEPERATED_POS_ROT 0
#define DUMP_3D_POSITIONS 0

unsigned int invisibleJoints=0;
unsigned int   visibleJoints=0;
unsigned int filteredOutCSVBehindPoses=0;
unsigned int filteredOutCSVOutPoses=0;
unsigned int filteredOutCSVPoses=0;

int fileExists(const char * filename)
{
 FILE *fp = fopen(filename,"r");
 if( fp ) { /* exists */ fclose(fp); return 1; }
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



int csvSkeletonFilter(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform * bvhTransform,
                       struct simpleRenderer * renderer,
                       unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                       unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                       unsigned int filterWeirdSkeletons
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

 return 1;
}



int dumpBVHToCSVHeader(
                       struct BVH_MotionCapture * mc,
                       const char * filename2D,
                       const char * filename3D,
                       const char * filenameBVH
                      )
{
   unsigned int jID=0;

   if (!fileExists(filename2D))
   {
    FILE * fp2D = fopen(filename2D,"a");

    if (fp2D!=0)
    {
     //2D Positions -------------------------------------------------------------------------------------------------------------
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
          fprintf(
                  fp2D,"2DX_%s,2DY_%s,visible_%s,",
                  mc->jointHierarchy[jID].jointName,
                  mc->jointHierarchy[jID].jointName,
                  mc->jointHierarchy[jID].jointName
                 );
         }
         else
         {
          unsigned int parentID=mc->jointHierarchy[jID].parentJoint;
          fprintf(
                  fp2D,"2DX_EndSite_%s,2DY_EndSite_%s,visible_EndSite_%s,",
                  mc->jointHierarchy[parentID].jointName,
                  mc->jointHierarchy[parentID].jointName,
                  mc->jointHierarchy[parentID].jointName
                 );
         }
       }
     //--------------------------------------------------------------------------------------------------------------------------
     fprintf(fp2D,"\n");
     fclose(fp2D);
   }
  }else
    {
     fprintf(stderr,"We don't need to regenerate the CSV 2D header, it already exists\n");
    }



   //3D Positions -------------------------------------------------------------------------------------------------------------
   if (!fileExists(filename3D))
   {
     FILE * fp3D = fopen(filename3D,"a");
     if (fp3D!=0)
     {
      for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           fprintf(fp3D,"3DX_%s,3DY_%s,3DZ_%s,",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName);
         }
       }
      fprintf(fp3D,"\n");
      fclose(fp3D);
     }
   } else
    {
     fprintf(stderr,"We don't need to regenerate the CSV 3D header, it already exists\n");
    }
   //--------------------------------------------------------------------------------------------------------------------------



   if (!fileExists(filenameBVH))
   {
     FILE * fpBVH = fopen(filenameBVH,"a");
     if (fpBVH!=0)
     {
     //Model Configuration
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           unsigned int channelID=0;
           for (channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
           {
            fprintf(
                    fpBVH,"%s_%s,",
                    mc->jointHierarchy[jID].jointName,
                    channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[channelID]]
                   );
           }
         }
       }
      //Append Frame ID
      fprintf(fpBVH,"\n");
      fclose(fpBVH);
     }
    } else
    {
     fprintf(stderr,"We don't need to regenerate the CSV header, it already exists\n");
    }
   //--------------------------------------------------------------------------------------------------------------------------



 return 1;
}



int dumpBVHToCSVBody(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform * bvhTransform,
                       struct simpleRenderer * renderer,
                       unsigned int fID,
                       const char * filename2D,
                       const char * filename3D,
                       const char * filenameBVH,
                       unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                       unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                       unsigned int filterWeirdSkeletons,
                       unsigned int encodeRotationsAsRadians
                      )
{
   unsigned int jID=0;

   if (
       !csvSkeletonFilter(
                           mc,
                           bvhTransform,
                           renderer,
                           filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                           filterOutSkeletonsWithAnyLimbsOutOfImage,
                           filterWeirdSkeletons
                         )
       )
   {
     return 0;
   }

   //-------------------------------------------------
   if (encodeRotationsAsRadians)
   {
    fprintf(stderr,"encodeRotationsAsRadians not implemented, please switch it off\n");
    exit(0);
   }//-----------------------------------------------

   FILE * fp2D = fopen(filename2D,"a");
   FILE * fp3D = fopen(filename3D,"a");
   FILE * fpBVH = fopen(filenameBVH,"a");

   unsigned int dumped=0;



     //2D Positions -------------------------------------------------------------------------------------------------------------
     if (fp2D!=0)
     {
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (bvhTransform->joint[jID].isOccluded) { ++invisibleJoints; } else { ++visibleJoints; }

         ///=================================================
         if (!mc->jointHierarchy[jID].isEndSite)
         {
             /*
          if (bvhTransform->joint[jID].isOccluded)
          {
           fprintf(fp,"0,0,0,");
          } else*/
          {
          fprintf(
                  fp2D,"%0.4f,%0.4f,%u,",
                  (float) bvhTransform->joint[jID].pos2D[0]/renderer->width,
                  (float) bvhTransform->joint[jID].pos2D[1]/renderer->height,
                  (bvhTransform->joint[jID].isOccluded==0)
                 );
          }
         }
         ///=================================================
         else
         {
          unsigned int parentID=mc->jointHierarchy[jID].parentJoint;
          /*
          if (bvhTransform->joint[jID].isOccluded)
          {
           fprintf(fp,"0,0,0,");
          } else*/
          {
              //jID parentID
          fprintf(
                  fp2D,"%0.4f,%0.4f,%u,",
                  (float) bvhTransform->joint[jID].pos2D[0]/renderer->width,
                  (float) bvhTransform->joint[jID].pos2D[1]/renderer->height,
                  (bvhTransform->joint[jID].isOccluded==0)
                 );
          }
         }
         ///=================================================
       }
     fprintf(fp2D,"\n");
     fclose(fp2D);
     ++dumped;
     }
     //-----------------------------------------------------------------------------------------------------------------------------


   //3D Positions -------------------------------------------
   if (fp3D!=0)
   {
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
          fprintf(
                  fp3D,"%0.2f,%0.2f,%0.2f,",
                  bvhTransform->joint[jID].pos3D[0],
                  bvhTransform->joint[jID].pos3D[1],
                  bvhTransform->joint[jID].pos3D[2]
                 );
         }
       }
     fprintf(fp3D,"\n");
     fclose(fp3D);
     ++dumped;
   }
   //-------------------------------------------------------------------


     //Joint Configuration
   if (fpBVH!=0)
   {
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           unsigned int channelID=0;
           for (channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
           {
             unsigned int channelType =  mc->jointHierarchy[jID].channelType[channelID];
             fprintf(
                     fpBVH,"%0.4f,",
                     bvh_getJointChannelAtFrame(mc,jID,fID,channelType)
                    );
           }
         }
       }
     fprintf(fpBVH,"\n");
     //-------------------------------------------------------------------
     fclose(fpBVH);
     ++dumped;
  }

 return (dumped==3);
}

