#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bvh_to_csv.h"
#include "bvh_export.h"

#include "../../TrajectoryParser/InputParser_C.h"

#include "../bvh_loader.h"
#include "../calculate/bvh_project.h"
#include "../edit/bvh_remapangles.h"
#include "../edit/bvh_cut_paste.h"

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
                       const char * filenameBVH,
                       const char * filenameMap
                      )
{
   if ( (filenameMap!=0) && (filenameMap[0]!=0) && (!bvhExportFileExists(filenameMap)) )
   {
    FILE * fpMap = fopen(filenameMap,"a");

    if (fpMap!=0)
    {
     fprintf(fpMap,"file,sample\n");
     fclose(fpMap);
    }
   }

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
                       const char * filenameMap,
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
   FILE * fpMap = 0;

   if ( (filename2D!=0)  && (filename2D[0]!=0) )   { fp2D  = fopen(filename2D,"a");   ++requestedToDump; }
   if ( (filename3D!=0)  && (filename3D[0]!=0) )   { fp3D  = fopen(filename3D,"a");   ++requestedToDump; }
   if ( (filenameBVH!=0) && (filenameBVH[0]!=0))   { fpBVH = fopen(filenameBVH,"a");  ++requestedToDump; }
   if ( (filenameMap!=0) && (filenameMap[0]!=0))   { fpMap = fopen(filenameMap,"a");  ++requestedToDump; }


   //Map File -------------------------------------------
   if (fpMap!=0)
   {
     fprintf(fpMap,"%s,%u\n",mc->fileName,fID);
     fclose(fpMap);
     ++dumped;
   }
   //-------------------------------------------------------------------




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



unsigned int countLinesInFile(const char *filename, size_t block_size)
{
    char buffer[block_size];
    unsigned int lineCount = 0;

    // Open the file in read mode
    FILE * file = fopen(filename, "r");

    // Check if file was opened successfully
    if (file != 0)
    {
     // Count lines in the file
     while (fgets(buffer, block_size, file) != NULL)
     {
        for (int i = 0; i < block_size && buffer[i] != '\0'; i++)
        {
            if (buffer[i] == '\n') { lineCount++; }
        }
     }
     // Close the file
     fclose(file);
    }

    return lineCount;
}




// ./BVHTester --from lhand.qbvh --importCSVPoses sobolLHand_131072.csv --csv ./ lhand_all.csv 2d+3d+bvh
int bvh_ImportCSVPoses(
                        struct BVH_MotionCapture * mc,
                        const char * filenameOfCSVFile
                      )
{
 int result = 0;
 int lineCount = countLinesInFile(filenameOfCSVFile,1024);
 fprintf(stderr,"File %s has %u lines \n",filenameOfCSVFile,lineCount);
  //-----------------------------------------------------------
  struct InputParserC * csvLine = InputParser_Create(1024,4);
  if (csvLine==0) { return 0; }
  //-----------------------------------------------------------
  InputParser_SetDelimeter(csvLine,0,',');
  InputParser_SetDelimeter(csvLine,1,'\t');
  InputParser_SetDelimeter(csvLine,2,10);
  InputParser_SetDelimeter(csvLine,3,13);
  //-----------------------------------------------------------
  char whereToStoreItems[512]={0};
  unsigned int numberOfHeaderParameters = 0;
  unsigned int * mID = 0;
  unsigned int fID = 0;
  unsigned int i;
  //-----------------------------------------------------------
  FILE * fp = fopen(filenameOfCSVFile,"r");
  if (fp!=0)
        {
            char * line = NULL;
            size_t len = 0;
            ssize_t read;

            unsigned int lineNumber=0;
            while ( (read = getline(&line, &len, fp)) != -1)
                {
                  if (line!=0)
                  {
                      if (lineNumber>0)
                      { //Reading a body line from CSV
                       if (lineNumber%10==0)
                        { fprintf(stderr,"\r   %s - Exporting Frame %u/%u %0.2f%%         \r",filenameOfCSVFile,lineNumber,lineCount,(float) (100*lineNumber)/lineCount); }

                          fID = lineNumber;
                          int numberOfRowParameters = InputParser_SeperateWordsCC(csvLine,line,1);
                          if (numberOfRowParameters!=numberOfHeaderParameters)
                          {
                           fprintf(stderr,"Incorrect number of parameters vs header..!\n");
                           break;
                          }

                          unsigned int mIDOffset = fID * mc->numberOfValuesPerFrame;
                          for (i=0; i<numberOfHeaderParameters; i++)
                           {
                               BVHMotionChannelID resolvedChannelID = mID[i] + mIDOffset;
                               mc->motionValues[resolvedChannelID]  = InputParser_GetWordFloat(csvLine,i);
                           }
                      } //Finished reading a body line from CSV
                        else
                      {
                       //Resolve CSV header to -> unsigned int * mID
                       fprintf(stderr,"Header %s \n",line);
                       numberOfHeaderParameters = InputParser_SeperateWordsCC(csvLine,line,1);
                       fprintf(stderr,"numberOfHeaderParameters %u \n",numberOfHeaderParameters);

                       if (!bvh_GrowMocapFileByCopyingExistingMotions(mc,lineCount-1))
                       {
                        fprintf(stderr,"Could not grow motion capture..!\n");
                        break;
                       }

                       mID = (unsigned int *) malloc(numberOfHeaderParameters * sizeof(unsigned int));
                       if (mID!=0)
                       {
                        for (i=0; i<numberOfHeaderParameters; i++)
                        {
                         InputParser_GetLowercaseWord(csvLine,i,whereToStoreItems,512);
                         fprintf(stderr,"Column %u / %s => ",i,whereToStoreItems);
                         unsigned int length = strlen(whereToStoreItems);
                         if (length>10)
                         {
                           char * jointName = whereToStoreItems;
                           char * dof   = whereToStoreItems;
                           if ( whereToStoreItems[length-10] == '_' )
                           {
                               whereToStoreItems[length-10] = 0;
                               dof = whereToStoreItems + (length-9);
                           } else
                           {
                               fprintf(stderr,"CSV file does not have the label format expected..!\n");
                               break;
                           }
                           fprintf(stderr,"Joint %s / DoF %s => ",jointName,dof);
                           //========================================================================
                           //Resolve degree of freedom..
                           //========================================================================
                           int channelID = BVH_CHANNEL_NONE;
                           if (strcmp(dof,"xposition")==0)  { channelID = BVH_POSITION_X;  } else
                           if (strcmp(dof,"yposition")==0)  { channelID = BVH_POSITION_Y;  } else
                           if (strcmp(dof,"zposition")==0)  { channelID = BVH_POSITION_Z;  } else
                           if (strcmp(dof,"wrotation")==0)  { channelID = BVH_ROTATION_W;  } else
                           if (strcmp(dof,"xrotation")==0)  { channelID = BVH_ROTATION_X;  } else
                           if (strcmp(dof,"yrotation")==0)  { channelID = BVH_ROTATION_Y;  } else
                           if (strcmp(dof,"zrotation")==0)  { channelID = BVH_ROTATION_Z;  } else
                           if (strcmp(dof,"xrodrigues")==0) { channelID = BVH_RODRIGUES_X; } else
                           if (strcmp(dof,"yrodrigues")==0) { channelID = BVH_RODRIGUES_Y; } else
                           if (strcmp(dof,"zrodrigues")==0) { channelID = BVH_RODRIGUES_Z; } else
                            {
                               fprintf(stderr,"Unknown degree of freedom (%s)..!\n",dof);
                               break;
                            }
                           //========================================================================

                           //========================================================================
                           //Resolve jointName -> jID..
                           //========================================================================
                           BVHJointID jID = 0;
                           fID = 0;
                           if ( bvh_getJointIDFromJointNameNocase(mc,jointName,&jID) )
                           {
                               fprintf(stderr,"Resolve jointID(%u)/channel(%u)..!\n",jID,channelID);
                               mID[i] = bvh_resolveFrameAndJointAndChannelToMotionID(mc,jID,fID,channelID);
                           } else
                           {
                               fprintf(stderr,"Unknown joint (%s)..!\n",jointName);
                               break;
                           }
                           //========================================================================
                         } //We have a big enough label to have xxxx_channelName
                        } //Parse each input column
                       } //We could allocate mID
                      } //End parsing header..

                      lineNumber+=1;
                  } //We have read a non-null line from the file
                }//We have read a line from the file
            fclose(fp);
            result = 1;
        } //We have opened a text file for reading
   //-----------------------------------------------------------
   InputParser_Destroy(csvLine);
   if (mID!=0) { free(mID); mID=0; }
   //-----------------------------------------------------------
   return result;
}

