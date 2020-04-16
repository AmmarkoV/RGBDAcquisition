#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bvh_to_csv.h"

#include "bvh_export.h"

#include "../bvh_loader.h"

#include "../calculate/bvh_project.h"
#include "../edit/bvh_remapangles.h"

#define CONVERT_EULER_TO_RADIANS M_PI/180.0
#define DUMP_SEPERATED_POS_ROT 0
#define DUMP_3D_POSITIONS 0


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */

/*
unsigned int invisibleJoints=0;
unsigned int   visibleJoints=0;
unsigned int filteredOutCSVBehindPoses=0;
unsigned int filteredOutCSVOutPoses=0;
unsigned int filteredOutCSVPoses=0;*/

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
                       struct filteringResults * filterStats,
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
           ++filterStats->filteredOutCSVPoses;
           ++filterStats->filteredOutCSVBehindPoses;
           //Just counting to reduce spam..
           //fprintf(stderr,"Filtered because of being behind camera..\n");
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
           ++filterStats->filteredOutCSVPoses;
           ++filterStats->filteredOutCSVOutPoses;
           //Just counting to reduce spam..
           //fprintf(stderr,"Filtered because of being limbs out of camera..\n");
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
           ++filterStats->filteredOutCSVPoses;
           return 0;
        }
   }//-----------------------------------------------

 return 1;
}


void considerIfJointIsSelected(
                                struct BVH_MotionCapture * mc,
                                unsigned int jID,
                                int * isJointSelected,
                                int * isJointEndSiteSelected
                               )
{
     *isJointSelected=1;
     *isJointEndSiteSelected=1;

     //First of all, if no joint selections have occured then everything is selected..
     if (!mc->selectedJoints) { return; }
      else
    {
     //If we reached this far it means there is a selection active..
     //We consider everything unselected unless proven otherwise..
     *isJointSelected=0;
     *isJointEndSiteSelected=0;


     //We now check if this joint is selected..
     //-------------------------------------------------------------
     //If there is a selection declared then let's consider if the joint is selected..

     if (mc->jointHierarchy[jID].isEndSite)
     {
       //If we are talking about an endsite we will have to check with it's parent joint..
       unsigned int parentID=mc->jointHierarchy[jID].parentJoint;
       if ( (mc->selectedJoints[parentID]) && (mc->selectionIncludesEndSites) )
                  { *isJointEndSiteSelected=1; }
     } else
     {
       //This is a regular joint..
       if (mc->selectedJoints[jID])
                          { *isJointSelected=1; }
     }

    }

}



int dumpBVHToCSVHeader(
                       struct BVH_MotionCapture * mc,
                       const char * filename2D,
                       const char * filename3D,
                       const char * filenameBVH
                      )
{
   unsigned int jID=0;
   int isJointSelected=1;
   int isJointEndSiteSelected=1;

   if ( (filename2D!=0) && (filename2D[0]!=0) && (!fileExists(filename2D)) )
   {
    FILE * fp2D = fopen(filename2D,"a");

    if (fp2D!=0)
    {
     char comma=' ';
     //2D Positions -------------------------------------------------------------------------------------------------------------
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
          considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);

          //----------------------------------
          //If we have hidden joints declared only the 2D part will be hidden..
          if (mc->hideSelectedJoints!=0)
            {  //If we want to hide the specific joint then it is not selected..
               if (mc->hideSelectedJoints[jID])
                  {
                     isJointSelected=0;
                     if (mc->hideSelectedJoints[jID]!=2) { isJointEndSiteSelected=0; }
                  }
            }
          //----------------------------------
         if (!mc->jointHierarchy[jID].isEndSite)
         {
            if (isJointSelected)
            {
                if (comma==',') { fprintf(fp2D,",");  } else { comma=','; }
                fprintf(fp2D,"2DX_%s,2DY_%s,visible_%s",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName);
            }
         }
         else
         {
            if (isJointEndSiteSelected)
            {
               unsigned int parentID=mc->jointHierarchy[jID].parentJoint;
               if (comma==',') { fprintf(fp2D,",");  } else { comma=','; }
               fprintf(fp2D,"2DX_EndSite_%s,2DY_EndSite_%s,visible_EndSite_%s",mc->jointHierarchy[parentID].jointName,mc->jointHierarchy[parentID].jointName,mc->jointHierarchy[parentID].jointName);
            }
         }
       }
     //--------------------------------------------------------------------------------------------------------------------------
     fprintf(fp2D,"\n");
     fclose(fp2D);
   }
  } else
  {
     fprintf(stderr,"We don't need to regenerate the CSV  header for 2D points, it already exists\n");
  }



   //3D Positions -------------------------------------------------------------------------------------------------------------
   if ( (filename3D!=0) && (filename3D[0]!=0) && (!fileExists(filename3D)) )
   {
     FILE * fp3D = fopen(filename3D,"a");
     if (fp3D!=0)
     {
      char comma=' ';

      for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);

         if (!mc->jointHierarchy[jID].isEndSite)
         {
            if (isJointSelected)
            {
                if (comma==',') { fprintf(fp3D,","); } else { comma=','; }
                fprintf(fp3D,"3DX_%s,3DY_%s,3DZ_%s",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName);
            }
         } else
         {
            if (isJointEndSiteSelected)
            {
             unsigned int parentID=mc->jointHierarchy[jID].parentJoint;
             if (comma==',') { fprintf(fp3D,","); } else { comma=','; }
             fprintf(fp3D,"3DX_EndSite_%s,3DY_EndSite_%s,3DZ_EndSite_%s",mc->jointHierarchy[parentID].jointName,mc->jointHierarchy[parentID].jointName,mc->jointHierarchy[parentID].jointName);
            }
         }
       }
      fprintf(fp3D,"\n");
      fclose(fp3D);
     }
   } else
    {
     fprintf(stderr,"We don't need to regenerate the CSV header for 3D Points, it already exists\n");
    }
   //--------------------------------------------------------------------------------------------------------------------------




   if ( (filenameBVH!=0) && (filenameBVH[0]!=0) && (!fileExists(filenameBVH)) )
   {
     FILE * fpBVH = fopen(filenameBVH,"a");
     if (fpBVH!=0)
     {

      //----------------------------------------------
      /*             IS THIS NEEDED ?
      unsigned int lastElement=0;
      for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           lastElement=jID;
         }
       }*/
      //----------------------------------------------



      char comma=' ';
      //Model Configuration
      for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
          considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);


         if (!mc->jointHierarchy[jID].isEndSite)
         {
            if (isJointSelected)
            {
                unsigned int channelID=0;
                for (channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
                 {
                    if (comma==',') { fprintf(fpBVH,",");  } else { comma=','; }
                    fprintf(
                                    fpBVH,"%s_%s",
                                    mc->jointHierarchy[jID].jointName,
                                   channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[channelID]]
                                  );
                 }
            }
         }
         //else
         //End Sites have no motion fields so they are not present here..

       }
      //Append Frame ID
      fprintf(fpBVH,"\n");
      fclose(fpBVH);
     }
    } else
    {
     fprintf(stderr,"We don't need to regenerate the CSV header for BVH motions, it already exists\n");
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
                       unsigned int csvOrientation,
                       struct filteringResults * filterStats,
                       unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                       unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                       unsigned int filterWeirdSkeletons,
                       unsigned int encodeRotationsAsRadians
                      )
{
   unsigned int jID=0;
   int isJointSelected=1;
   int isJointEndSiteSelected=1;

   if (
       !csvSkeletonFilter(
                           mc,
                           bvhTransform,
                           renderer,
                           filterStats,
                           filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                           filterOutSkeletonsWithAnyLimbsOutOfImage,
                           filterWeirdSkeletons
                         )
       )
   {
     //fprintf(stderr,"csvSkeletonFilter discarded frame %u\n",fID);
     return 0;
   }

   //-------------------------------------------------
   if (encodeRotationsAsRadians)
   {
    fprintf(stderr,"encodeRotationsAsRadians not implemented, please switch it off\n");
    exit(0);
   }//-----------------------------------------------


   unsigned int dumped=0;
   unsigned int requestedToDump=0;
   FILE * fp2D = 0;
   FILE * fp3D = 0;
   FILE * fpBVH = 0;

   if ( (filename2D!=0) && (filename2D[0]!=0) )   { fp2D = fopen(filename2D,"a");   ++requestedToDump; }
   if ( (filename3D!=0) && (filename3D[0]!=0) )   { fp3D = fopen(filename3D,"a");   ++requestedToDump; }
   if ( (filenameBVH!=0) && (filenameBVH[0]!=0) ) { fpBVH = fopen(filenameBVH,"a"); ++requestedToDump; }


     //--------------------------------------------------------------------------------------------------------------------------
     //---------------------------------------------------2D Positions ----------------------------------------------------------
     //--------------------------------------------------------------------------------------------------------------------------
     if (fp2D!=0)
     {
      char comma=' ';
      for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
          considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
          //----------------------------------
          //If we have hidden joints declared only the 2D part will be hidden..
          if (mc->hideSelectedJoints!=0)
            {  //If we want to hide the specific joint then it is not selected..
               if (mc->hideSelectedJoints[jID])
                {
                  isJointSelected=0;
                  if (mc->hideSelectedJoints[jID]!=2) { isJointEndSiteSelected=0; }
                }
            }
          //----------------------------------

         if (
               //If this a regular joint and regular joints are enabled
               ( (!mc->jointHierarchy[jID].isEndSite) && (isJointSelected) )
                    ||
               //OR if this is an end joint and end joints are enabled..
               ( (mc->jointHierarchy[jID].isEndSite) && (isJointEndSiteSelected) )
            )
          {
                if (bvhTransform->joint[jID].isOccluded) { ++filterStats->invisibleJoints; } else { ++filterStats->visibleJoints; }

                if (mc->jointHierarchy[jID].erase2DCoordinates)
                    {
                       if (comma==',') { fprintf(fp2D,",");  } else { comma=','; }
                        fprintf(fp2D,"0,0,0");
                    } else
                    {
                       if (comma==',') { fprintf(fp2D,",");  } else { comma=','; }
                       fprintf(
                               fp2D,"%0.6f,%0.6f,%u",
                               (float) bvhTransform->joint[jID].pos2D[0]/renderer->width,
                               (float) bvhTransform->joint[jID].pos2D[1]/renderer->height,
                               (bvhTransform->joint[jID].isOccluded==0)
                               );
                    }
         }
       }
     fprintf(fp2D,"\n");
     fclose(fp2D);
     ++dumped;
     }
     //-----------------------------------------------------------------------------------------------------------------------------
     //-----------------------------------------------------------------------------------------------------------------------------
     //-----------------------------------------------------------------------------------------------------------------------------


   //3D Positions -------------------------------------------
   if (fp3D!=0)
   {
     char comma=' ';
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);

         if (!mc->jointHierarchy[jID].isEndSite)
         {
             if (isJointSelected)
             {
               if (comma==',') { fprintf(fp3D,",");  } else { comma=','; }
                       fprintf(
                                       fp3D,"%f,%f,%f",bvhTransform->joint[jID].pos3D[0],bvhTransform->joint[jID].pos3D[1],bvhTransform->joint[jID].pos3D[2]
                                      );
             }
         } else
         {
             if (isJointEndSiteSelected)
             {
               if (comma==',') { fprintf(fp3D,",");  } else { comma=','; }
                fprintf(
                                fp3D,"%f,%f,%f",bvhTransform->joint[jID].pos3D[0],bvhTransform->joint[jID].pos3D[1],bvhTransform->joint[jID].pos3D[2]
                              );
             }
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
     char comma=' ';
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
          considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);


         if ( (!mc->jointHierarchy[jID].isEndSite) && (isJointSelected) )
         {
           unsigned int channelID=0;
           for (channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
           {
             unsigned int channelType =  mc->jointHierarchy[jID].channelType[channelID];

             float value = bvh_getJointChannelAtFrame(mc,jID,fID,channelType);

             //Due to the particular requirements of MocapNET we need to be able to split orientations in CSV files..
             //We want the neural network to only work with values normalized and centered around 0
             if (csvOrientation!=BVH_ENFORCE_NO_ORIENTATION)
             {
              //TODO: add here a check for hip Y rotation and perform orientation change..
              if ( (jID==0) && (channelID==BVH_POSITION_X) ) //BVH_ROTATION_X
              {
                  //Test using :
                  //rm tmp/bvh_test.csv tmp/2d_test.csv && ./BVHTester --from Motions/MotionCapture/01/01_02.bvh  --repeat 0 --selectJoints 17 hip abdomen chest neck head rshoulder relbow rhand lshoulder lelbow lhand rhip rknee rfoot lhip lknee lfoot --csvOrientation right --randomize2D 1000 5000 -35 45 -35 35 135 35 --occlusions --csv tmp test.csv 2d+bvh
                  //value=666; <- highlight the correct
                  //value=(float) bvh_constrainAngleCentered0((double) value,0);
                  value=(float) bvh_RemapAngleCentered0((double) value,csvOrientation);
              }
             }

             if (comma==',') { fprintf(fpBVH,",");  } else { comma=','; }
             fprintf(fpBVH,"%0.5f",value);
           }
         }
         //else
         //BVH End Sites have no motion parameters so they dont need to be considered here..

       }
     fprintf(fpBVH,"\n");
     //-------------------------------------------------------------------
     fclose(fpBVH);
     ++dumped;
  }
   //-------------------------------------------------------------------

 //fprintf(stderr,"Dumped %u , Requested to Dump %u \n",dumped,requestedToDump);
 return (dumped==requestedToDump);
}

