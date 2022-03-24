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
                        int wiped2DOutput,
                        int wiped3DOutput,
                        int wipedBVHOutput,
                        int * did2DOutputPreExist,
                        int * did3DOutputPreExist,
                        int * didBVHOutputPreExist,
                        const char * filenameInput,
                        const char * filename3D,
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
   //fprintf(stderr,"dumpBVHToJSON(in=%s,out=%s)\n",filenameInput,filenameBVH);
   int isJointSelected=1;
   int isJointEndSiteSelected=1;
   unsigned int countedNumberOfJoints = countNumberOfJoints(mc);


   if  ( (filenameInput!=0) && (filenameInput[0]!=0) && ( (wiped2DOutput) || (!bvhExportFileExists(filenameInput)) ) )
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

     fprintf(fp,"\"2DJointNames\":\n");
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

     fprintf(fp,"\"2DJointSamples\":\n");
     fprintf(fp,"  [\n");
     fclose(fp);

     //For sure no 2D output pre-exists now!
     *did2DOutputPreExist = 0;
    } //We managed to open the file
   } else
   {
     fprintf(stderr,"We don't need to regenerate the JSON header for 2D points, it already exists\n");
   } //We have a filename..









   if ( (filename3D!=0) && (filename3D[0]!=0) && ( (wiped3DOutput) || (!bvhExportFileExists(filename3D)) ) )
   {
     FILE * fp3D = fopen(filename3D,"a");
     if (fp3D!=0)
     {
      fprintf(fp3D,"{\n");
      //Give camera information here !
      fprintf(fp3D,"\"VirtualCameraName\":\"Default Camera\",\n");
      fprintf(fp3D,"\"FocalLengthX\":%f,\n",fx);
      fprintf(fp3D,"\"FocalLengthY\":%f,\n",fy);
      fprintf(fp3D,"\"CenterX\":%f,\n",cx);
      fprintf(fp3D,"\"CenterY\":%f,\n",cy);
      fprintf(fp3D,"\"Width\":%f,\n",width);
      fprintf(fp3D,"\"Height\":%f,\n",height);
      fprintf(fp3D,"\"Near\":%f,\n",near);
      fprintf(fp3D,"\"Far\":%f,\n",far);
      fprintf(fp3D,"\"SelfSupervised\":0,\n");
      fprintf(fp3D,"\"Perturbed\":0,\n");
      fprintf(fp3D,"\"NumberOfJoints\":%u,\n",countedNumberOfJoints);

      fprintf(fp3D,"\"3DJointNames\":\n");
      fprintf(fp3D,"  [\n");

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
               if (comma==',') { fprintf(fp3D,",");  } else { comma=','; }
               fprintf(fp3D,"    {\n");
               fprintf(fp3D,"     \"Joint\" : \"%s\",\n",mc->jointHierarchy[jID].jointName);
               fprintf(fp3D,"     \"JointID\" : %u\n",jID);
               fprintf(fp3D,"    }\n");
            }
         }
         else
         {
            if (isJointEndSiteSelected)
            {
               unsigned int parentID=mc->jointHierarchy[jID].parentJoint;
               if (comma==',') { fprintf(fp3D,",");  } else { comma=','; }
               fprintf(fp3D,"    {\n");
               fprintf(fp3D,"     \"Joint\" : \"EndSite_%s\",\n",mc->jointHierarchy[parentID].jointName);
               fprintf(fp3D,"     \"JointID\" : %u\n",jID);
               fprintf(fp3D,"    }\n");
            }
         }
       }

      //Append Frame ID
      fprintf(fp3D," ],\n\n");
      fprintf(fp3D,"\"3DJointSamples\":\n");
      fprintf(fp3D,"  [\n");

      fclose(fp3D);


     //For sure no 3D output pre-exists now!
     *did3DOutputPreExist = 0;
     }
    } else
    {
     fprintf(stderr,"We don't need to regenerate the JSON header for 3D data, it already exists\n");
    }
   //--------------------------------------------------------------------------------------------------------------------------













   if ( (filenameBVH!=0) && (filenameBVH[0]!=0) && ( (wipedBVHOutput) || (!bvhExportFileExists(filenameBVH)) ) )
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
                    fprintf(fpBVH,"     \"JointID\" : %u,\n",jID);
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


      //For sure no 3D output pre-exists now!
      *didBVHOutputPreExist = 0;
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
                        const char * filename3D,
                        const char * filenameBVH
                       )
{
   if ( (filenameInput!=0) && (filenameInput[0]!=0) && (!bvhExportFileExists(filenameInput)) )
   {
    FILE * fp = fopen(filenameInput,"a");

    if (fp!=0)
    {
      fprintf(fp,"] \n }\n");
      fclose(fp);
     }
    }


   if ( (filename3D!=0) && (filename3D[0]!=0) && (!bvhExportFileExists(filename3D)) )
   {
     FILE * fp3D = fopen(filename3D,"a");
     if (fp3D!=0)
     {
      fprintf(fp3D,"] \n }\n");
      fclose(fp3D);
     }
   }


   if ( (filenameBVH!=0) && (filenameBVH[0]!=0) && (!bvhExportFileExists(filenameBVH)) )
   {
     FILE * fpBVH = fopen(filenameBVH,"a");
     if (fpBVH!=0)
     {
      fprintf(fpBVH,"] \n }\n");
      fclose(fpBVH);
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
                       const char * filename3D,
                       const char * filenameBVH,
                       int didInputOutputPreExist,
                       int did3DOutputPreExist,
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

   //-------------------------------------------------
   if (encodeRotationsAsRadians)
   {
    fprintf(stderr,"encodeRotationsAsRadians not implemented, please switch it off\n");
    exit(0);
   }//-----------------------------------------------


   unsigned int dumped=0;
   unsigned int requestedToDump=0;
   //-------------------------------------------------------------------------------------------------------------
   FILE * fp    = 0;
   FILE * fp3D  = 0;
   FILE * fpBVH = 0;
   //-------------------------------------------------------------------------------------------------------------
   if ( (filenameInput!=0) && (filenameInput[0]!=0) )   { fp = fopen(filenameInput,"a");     ++requestedToDump; }
   if ( (filename3D!=0)    && (filename3D[0]!=0) )      { fp3D = fopen(filename3D,"a");      ++requestedToDump; }
   if ( (filenameBVH!=0)   && (filenameBVH[0]!=0) )     { fpBVH = fopen(filenameBVH,"a");    ++requestedToDump; }
   //-------------------------------------------------------------------------------------------------------------


   //--------------------------------------------------------------------------------------------------------------------------
   //---------------------------------------------------2D Positions ----------------------------------------------------------
   //--------------------------------------------------------------------------------------------------------------------------
   if (fp!=0)
     {
      if (didInputOutputPreExist) { fprintf(fp,","); }
      fprintf(fp,"[");
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
                       //Please note that our 2D input is normalized [0..1]
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







   if (fp3D!=0)
     {
      if (did3DOutputPreExist) { fprintf(fp3D,","); }
      fprintf(fp3D,"[");
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
            if (comma==',') { fprintf(fp3D,",");  } else { comma=','; }
            //Please note that our 3D positions are stored in their native "scale" straight out of the renderer
            //They depend on the BVH scale/units (expect if the --scale argument ( or scaleWorld argument on bvh_loadBVH )
            //has altered them, that being said in case of MocapNET / they should be centimeters
            fprintf(
                     fp3D,"{\"x\":%f,\"y\":%f,\"z\":%f}",
                     bvhTransform->joint[jID].pos3D[0],
                     bvhTransform->joint[jID].pos3D[1],
                     bvhTransform->joint[jID].pos3D[2]
                    );
         }
       }
     fprintf(fp3D,"]\n");
     fclose(fp3D);
     ++dumped;
     }










   //Joint Configuration
   if (fpBVH!=0)
   {
     char comma=' ';
     if (didBVHOutputPreExist) { fprintf(fpBVH,","); }
     fprintf(fpBVH,"[");
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
