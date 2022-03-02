#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bvh_to_json.h"

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

int countNumberOfJoints(struct BVH_MotionCapture * mc)
{
 int isJointSelected=1;
 int isJointEndSiteSelected=1;
 unsigned int numberOfJoints = 0;
     for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
       {
          bvh_considerIfJointIsSelected(mc,jID,&isJointSelected,&isJointEndSiteSelected);
         //-----------------------------------------------------------------------------
          if (mc->hideSelectedJoints!=0)
            {  //If we want to hide the specific joint then it is not selected..
               if (mc->hideSelectedJoints[jID])
                  {
                     isJointSelected=0;
                     if (mc->hideSelectedJoints[jID]!=2) { isJointEndSiteSelected=0; }
                  }
            }
          //----------------------------------
         if (!mc->jointHierarchy[jID].isEndSite) { if (isJointSelected) { ++numberOfJoints; } } else
                                                 { if (isJointEndSiteSelected) { ++numberOfJoints; } }
       }
  return numberOfJoints;
}



int dumpBVHToJSONHeader(
                        struct BVH_MotionCapture * mc,
                        const char * filenameInput,
                        const char * filenameBVH,
                        float fx,
                        float fy,
                        float cx,
                        float cy,
                        float near,
                        float far,
                        float width,
                        float height
                       )
{
   fprintf(stderr,"dumpBVHToJSON(in=%s,out=%s)\n",filenameInput,filenameBVH);
   int isJointSelected=1;
   int isJointEndSiteSelected=1;
   unsigned int countedNumberOfJoints = countNumberOfJoints(mc);


   if ( (filenameInput!=0) && (filenameInput[0]!=0) && (!bvhExportFileExists(filenameInput)) )
   {
    FILE * fp = fopen(filenameInput,"a");

    if (fp!=0)
    {
     fprintf(fp,"{\n");
     //Give camera information here !
     fprintf(fp,"\"VirtualCameraName\":\"Default Camera\",\n");
     fprintf(fp,"\"FocalLengthX\":%f,\n",fx);
     fprintf(fp,"\"FocalLengthY\":%f,\n",fy);
     fprintf(fp,"\"CenterX\":%f,\n",cx);
     fprintf(fp,"\"CenterY\":%f,\n",cy);
     fprintf(fp,"\"Width\":%f,\n",width);
     fprintf(fp,"\"Height\":%f,\n",height);
     fprintf(fp,"\"Near\":%f,\n",near);
     fprintf(fp,"\"Far\":%f,\n",far);
     fprintf(fp,"\"SelfSupervised\":0,\n");
     fprintf(fp,"\"Perturbed\":0,\n");
     fprintf(fp,"\"NumberOfJoints\":%u,\n",countedNumberOfJoints);

     fprintf(fp,"\"2DJointsNames\":\n");
     fprintf(fp,"  [\n");

     char comma=' ';
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
               if (comma==',') { fprintf(fp,",");  } else { comma=','; }
               fprintf(fp,"    {\n");
               fprintf(fp,"     \"Joint\" : \"%s\",\n",mc->jointHierarchy[jID].jointName);
               fprintf(fp,"     \"JointID\" : %u\n",jID);
               fprintf(fp,"    }\n");
            }
         }
         else
         {
            if (isJointEndSiteSelected)
            {
               unsigned int parentID=mc->jointHierarchy[jID].parentJoint;
               if (comma==',') { fprintf(fp,",");  } else { comma=','; }
               fprintf(fp,"    {\n");
               fprintf(fp,"     \"Joint\" : \"EndSite_%s\",\n",mc->jointHierarchy[parentID].jointName);
               fprintf(fp,"     \"JointID\" : %u\n",jID);
               fprintf(fp,"    }\n");
            }
         }
       }
     fprintf(fp,"  ],\n\n");

     fprintf(fp,"\"2DJoints\":\n");
     fprintf(fp,"  [\n");
     fclose(fp);
    } //We managed to open the file
   } else
   {
     fprintf(stderr,"We don't need to regenerate the JSON header for 2D points, it already exists\n");
   } //We have a filename..





   if ( (filenameBVH!=0) && (filenameBVH[0]!=0) && (!bvhExportFileExists(filenameBVH)) )
   {
     FILE * fpBVH = fopen(filenameBVH,"a");
     if (fpBVH!=0)
     {
      fprintf(fpBVH,"{\n");
      //Give camera information here !
      fprintf(fpBVH,"\"VirtualCameraName\":\"Default Camera\",\n");
      fprintf(fpBVH,"\"FocalLengthX\":%f,\n",fx);
      fprintf(fpBVH,"\"FocalLengthY\":%f,\n",fy);
      fprintf(fpBVH,"\"CenterX\":%f,\n",cx);
      fprintf(fpBVH,"\"CenterY\":%f,\n",cy);
      fprintf(fpBVH,"\"Width\":%f,\n",width);
      fprintf(fpBVH,"\"Height\":%f,\n",height);
      fprintf(fpBVH,"\"Near\":%f,\n",near);
      fprintf(fpBVH,"\"Far\":%f,\n",far);
      fprintf(fpBVH,"\"SelfSupervised\":0,\n");
      fprintf(fpBVH,"\"Perturbed\":0,\n");
      fprintf(fpBVH,"\"NumberOfJoints\":%u,\n",countedNumberOfJoints);


      fprintf(fpBVH,"\"BVHJointChannelNames\":\n");
      fprintf(fpBVH,"  [\n");

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
                    fprintf(fpBVH,"    {\n");
                    fprintf(fpBVH,"     \"Joint\" : \"%s_%s\",\n", mc->jointHierarchy[jID].jointName,channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[channelID]]);
                    fprintf(fpBVH,"     \"JointID\" : %u\n",jID);
                    fprintf(fpBVH,"     \"ChannelID\" : %u\n",channelID);
                    fprintf(fpBVH,"    }\n");
                 }
         }//End Sites have no motion fields so they are not present here..
       }

      //Append Frame ID
      fprintf(fpBVH," ],\n\n");


      fprintf(fpBVH,"\"BVHChannels\":\n");
      fprintf(fpBVH,"  [\n");

      fclose(fpBVH);
     }
    } else
    {
     fprintf(stderr,"We don't need to regenerate the JSON header for BVH motions, it already exists\n");
    }
   //--------------------------------------------------------------------------------------------------------------------------

 return 1;
}





int dumpBVHToJSONFooter(
                        struct BVH_MotionCapture * mc,
                        const char * filenameInput,
                        const char * filenameBVH
                       )
{
   if ( (filenameInput!=0) && (filenameInput[0]!=0) && (!bvhExportFileExists(filenameInput)) )
   {
    FILE * fp = fopen(filenameInput,"a");

    if (fp!=0)
    {
     fprintf(fp,"] \n }\n");
     }
    }



   if ( (filenameBVH!=0) && (filenameBVH[0]!=0) && (!bvhExportFileExists(filenameBVH)) )
   {
     FILE * fpBVH = fopen(filenameBVH,"a");
     if (fpBVH!=0)
     {
      fprintf(fpBVH,"] \n }\n");
     }
   }

  return 1;
}





int dumpBVHToJSONBody(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform     * bvhTransform,
                       struct simpleRenderer    * renderer,
                       unsigned int fID,
                       const char * filenameInput,
                       const char * filenameBVH,
                       int didInputOutputPreExist,
                       int didBVHOutputPreExist,
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
   FILE * fp = 0;
   FILE * fpBVH = 0;

   if ( (filenameInput!=0) && (filenameInput[0]!=0) )   { fp = fopen(filenameInput,"a");     ++requestedToDump; }
   if ( (filenameBVH!=0)   && (filenameBVH[0]!=0) )     { fpBVH = fopen(filenameBVH,"a");    ++requestedToDump; }


   //--------------------------------------------------------------------------------------------------------------------------
   //---------------------------------------------------2D Positions ----------------------------------------------------------
   //--------------------------------------------------------------------------------------------------------------------------
   if (fp!=0)
     {
      if (didInputOutputPreExist) { fprintf(fp,","); }
      fprintf(fp,"  [");
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
                       if (comma==',') { fprintf(fp,",");  } else { comma=','; }
                       fprintf(fp,"{\"x\":0,\"y\":0,\"v\":0}");
                    } else
                    {
                       if (comma==',') { fprintf(fp,",");  } else { comma=','; }
                       fprintf(
                               fp,"{\"x\":%0.6f,\"y\":%0.6f,\"v\":%u}",
                               (float) bvhTransform->joint[jID].pos2D[0]/renderer->width,
                               (float) bvhTransform->joint[jID].pos2D[1]/renderer->height,
                               (bvhTransform->joint[jID].isOccluded==0)
                              );
                    }
         }
       }
     fprintf(fp,"]\n");
     fclose(fp);
     ++dumped;
     }
   //-----------------------------------------------------------------------------------------------------------------------------
   //-----------------------------------------------------------------------------------------------------------------------------
   //-----------------------------------------------------------------------------------------------------------------------------

   //Joint Configuration
   if (fpBVH!=0)
   {
     char comma=' ';
     if (didBVHOutputPreExist) { fprintf(fpBVH,","); }
     fprintf(fpBVH," [");
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

             if (comma==',') { fprintf(fpBVH,",");  } else { comma=','; }
             fprintf(fpBVH,"{\"v\":%0.5f}",value);
           }
         }
         //else
         //BVH End Sites have no motion parameters so they dont need to be considered here..
       }
     fprintf(fpBVH,"]\n");
     //-------------------------------------------------------------------
     fclose(fpBVH);
     ++dumped;
  }
   //-------------------------------------------------------------------

 //fprintf(stderr,"Dumped %u , Requested to Dump %u \n",dumped,requestedToDump);
 return (dumped==requestedToDump);
}
