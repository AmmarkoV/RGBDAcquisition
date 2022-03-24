#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bvh_to_csv.h"
#include "bvh_export.h"

#include "../bvh_loader.h"
#include "../calculate/bvh_project.h"
#include "../edit/bvh_remapangles.h"

#define DUMP_SEPERATED_POS_ROT 0
#define DUMP_3D_POSITIONS 0

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */




int dumpBVHToCSVHeader(
                       struct BVH_MotionCapture * mc,
                       const char * filename2D,
                       const char * filename3D,
                       const char * filenameBVH
                      )
{
   int isJointSelected=1;
   int isJointEndSiteSelected=1;

   if ( (filename2D!=0) && (filename2D[0]!=0) && (!bvhExportFileExists(filename2D)) )
   {
    FILE * fp2D = fopen(filename2D,"a");

    if (fp2D!=0)
    {
     char comma=' ';
     //2D Positions -------------------------------------------------------------------------------------------------------------
     for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
       {
          bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
         //-----------------------------------------------------------------------------

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
   if ( (filename3D!=0) && (filename3D[0]!=0) && (!bvhExportFileExists(filename3D)) )
   {
     FILE * fp3D = fopen(filename3D,"a");
     if (fp3D!=0)
     {
      char comma=' ';

      for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
       {
         bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
         //-----------------------------------------------------------------------------
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


   if ( (filenameBVH!=0) && (filenameBVH[0]!=0) && (!bvhExportFileExists(filenameBVH)) )
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
      for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
       {
          bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
         //-----------------------------------------------------------------------------
         if ( (!mc->jointHierarchy[jID].isEndSite) && (isJointSelected) )
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
                       struct filteringResults * filterStats,
                       unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                       unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                       unsigned int filterWeirdSkeletons,
                       unsigned int encodeRotationsAsRadians
                      )
{
   int isJointSelected=1;
   int isJointEndSiteSelected=1;

   if (
       !bvhExportSkeletonFilter(
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
      for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
       {
          bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
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
               ( (!mc->jointHierarchy[jID].isEndSite) && (isJointSelected) ) ||
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

                       //Please note that our 2D input is normalized [0..1]
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
     for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
       {
         bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
         //-----------------------------------------------------------------------------
         if (
               ( (!mc->jointHierarchy[jID].isEndSite) && (isJointSelected) )        ||
               ( (mc->jointHierarchy[jID].isEndSite) && (isJointEndSiteSelected) )
            )
         {
           if (comma==',') { fprintf(fp3D,",");  } else { comma=','; }

           //Please note that our 3D positions are stored in their native "scale" straight out of the renderer
           //They depend on the BVH scale/units (expect if the --scale argument ( or scaleWorld argument on bvh_loadBVH )
           //has altered them, that being said in case of MocapNET / they should be centimeters
           fprintf(
                    fp3D,"%f,%f,%f",
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
     char comma=' ';
     for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
       {
         bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
         //-----------------------------------------------------------------------------
         if ( (!mc->jointHierarchy[jID].isEndSite) && (isJointSelected) )
         {
           unsigned int channelID=0;
           for (channelID=0; channelID<mc->jointHierarchy[jID].loadedChannels; channelID++)
           {
             unsigned int channelType =  mc->jointHierarchy[jID].channelType[channelID];

             float value = bvh_getJointChannelAtFrame(mc,jID,fID,channelType);
/*           //NO LONGER NEEDED, neural networks work fine without being centered @ 0!
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
                  //value=(float) bvh_constrainAngleCentered0((float) value,0);
                  value=(float) bvh_RemapAngleCentered0((float) value,csvOrientation);
              }
             }  */
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

