#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bvh_export.h"
#include "bvh_to_json.h"
#include "bvh_to_csv.h"
#include "bvh_to_svg.h"

#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */

#define CONVERT_EULER_TO_RADIANS M_PI/180.0

int bvhExportFileExists(const char * filename)
{
 FILE *fp = fopen(filename,"r");
 if( fp ) { /* exists */ fclose(fp); return 1; }
 return 0;
}


int bvhExportFileWipe(const char * filename)
{
 FILE *fp = fopen(filename,"w");
 if( fp ) { /* exists */ fclose(fp); return 1; }
 return 0;
}


float bvhExportEulerAngleToRadiansIfNeeded( float eulerAngle , unsigned int isItNeeded)
{
  if (isItNeeded)
  {
    return eulerAngle*CONVERT_EULER_TO_RADIANS;
  }
 return eulerAngle;
}

int bvhExportSkeletonFilter(
                            struct BVH_MotionCapture * mc,
                            struct BVH_Transform * bvhTransform,
                            struct simpleRenderer * renderer,
                            struct filteringResults * filterStats,
                            unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                            unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                            unsigned int filterWeirdSkeletons
                           )
{
   //-------------------------------------------------
   if (filterOutSkeletonsWithAnyLimbsBehindTheCamera)
   {
     for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
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
     for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
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
       for (unsigned int jID=0; jID<mc->jointHierarchySize; jID++)
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


int performPointProjectionsForFrameForcingPositionAndRotation(
                                                              struct BVH_MotionCapture * mc,
                                                              struct BVH_Transform * bvhTransform,
                                                              unsigned int fID,
                                                              struct simpleRenderer * renderer,
                                                              float * forcePosition,
                                                              float * forceRotation,
                                                              unsigned int occlusions,
                                                              unsigned int directRendering
                                                             )
{
   BVHJointID rootJoint=0;

   if (
        !bvh_getRootJointID(
                             mc,
                             &rootJoint
                           )
      )
      {
        fprintf(stderr,RED "Error accessing root joint for frame %u\n" NORMAL,fID);
        return 0;
      }

   float dataOriginal[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER]={0};
   if (!bhv_populatePosXYZRotXYZ(mc,rootJoint,fID,dataOriginal,sizeof(dataOriginal)))
      {
        fprintf(stderr,RED "Error accessing original position/rotation data for frame %u\n" NORMAL,fID);
        return 0;
      }

   float dataOur[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_NUMBER]={0};
   dataOur[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_X]=forcePosition[0];
   dataOur[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Y]=forcePosition[1];
   dataOur[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_POSITION_Z]=forcePosition[2];
   dataOur[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_X]=forceRotation[0];
   dataOur[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Y]=forceRotation[1];
   dataOur[MOTIONBUFFER_TRANSACTION_DATA_FIELDS_ROTATION_Z]=forceRotation[2];

   if (mc->jointHierarchy[rootJoint].hasQuaternionRotation)
   {
        fprintf(stderr,RED "performPointProjectionsForFrameForcingPositionAndRotation: Cannot handle quaternions..\n" NORMAL);
        return 0;
   }

   if (!bhv_setPosXYZRotXYZ(mc,rootJoint,fID,dataOur,sizeof(dataOur)))
      {
        fprintf(stderr,RED "Error adjusting position/rotation data for frame %u\n" NORMAL,fID);
        return 0;
      }

  //Try to load the 3D positions of each joint for this particular frame..
   if (bvh_loadTransformForFrame(mc,fID,bvhTransform,1/*Populate extra structures*/))
       {
        //If we succeed then we can perform the point projections to 2D..
        //Project 3D positions on 2D frame and save results..

        if (!bhv_setPosXYZRotXYZ(mc,rootJoint,fID,dataOriginal,sizeof(dataOriginal)))
        {
         fprintf(stderr,RED "Error restoring original data for frame %u\n" NORMAL,fID);
         return 0;
        }

        return bvh_projectTo2D(mc,bvhTransform,renderer,occlusions,directRendering);
       } else
       //If we fail to load transform , then we can't do any projections and need to clean up
       {
         bvh_cleanTransform(mc,bvhTransform);
       }
   //----------------------------------------------------------
 return 0;
}



int performPointProjectionsForFrame(
                                     struct BVH_MotionCapture * mc,
                                     struct BVH_Transform * bvhTransform,
                                     unsigned int fID,
                                     struct simpleRenderer * renderer,
                                     unsigned int occlusions,
                                     unsigned int directRendering
                                    )
{
  //Try to load the 3D positions of each joint for this particular frame..
   if (bvh_loadTransformForFrame(mc,fID,bvhTransform,1/*Populate extra structures*/))
       {
        //If we succeed then we can perform the point projections to 2D..
        //Project 3D positions on 2D frame and save results..
        return bvh_projectTo2D(mc,bvhTransform,renderer,occlusions,directRendering);
       } else
       //If we fail to load transform , then we can't do any projections and need to clean up
       {
           fprintf(stderr,RED "performPointProjectionsForFrame: Could not load transform for frame %u\n" NORMAL,fID);
           bvh_cleanTransform(mc,bvhTransform);
       }
   //----------------------------------------------------------
 return 0;
}

int performPointProjectionsForMotionBuffer(
                                            struct BVH_MotionCapture * mc,
                                            struct BVH_Transform * bvhTransform,
                                            float * motionBuffer,
                                            struct simpleRenderer * renderer,
                                            unsigned int occlusions,
                                            unsigned int directRendering
                                           )
{
  //First load the 3D positions of each joint from a motion buffer (instead of a frame [see performPointProjectionsForFrame]..
  if (bvh_loadTransformForMotionBuffer(mc,motionBuffer,bvhTransform,1/*Populate extra structures*/))
       {
        //If we succeed then we can perform the point projections to 2D..
        //Project 3D positions on 2D frame and save results..
         return bvh_projectTo2D(mc,bvhTransform,renderer,occlusions,directRendering);
       } else
       //If we fail to load transform , then we can't do any projections and need to clean up
       { bvh_cleanTransform(mc,bvhTransform); }
   //----------------------------------------------------------
 return 0;
}


int dumpBVHTo_JSON_SVG_CSV(
                           const char * directory,
                           const char * filename,
                           int convertToJSON,
                           int convertToSVG,
                           int convertToCSV,
                           int convertToAngleHeatmap,
                           int useCSV_2D_Output,int useCSV_3D_Output,int useCSV_BVH_Output,
                           int wipe_2D_Output,int wipe_3D_Output,int wipe_BVH_Output,
                           struct BVH_MotionCapture * mc,
                           struct BVH_RendererConfiguration * renderConfig,
                           struct filteringResults * filterStats,
                           unsigned int sampleSkip,
                           unsigned int occlusions,
                           unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                           unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                           unsigned int filterWeirdSkeletons,
                           unsigned int encodeRotationsAsRadians
                          )
{
  struct BVH_Transform bvhTransform={0};

  char svgFilename[512];
  //Declare and populate csv output files, we have 2D,3D and BVH files...
  char csvFilename2D[512]={0};
  char csvFilename3D[512]={0};
  char csvFilenameBVH[512]={0};
  if (useCSV_2D_Output)  { snprintf(csvFilename2D,512,"%s/2d_%s",directory,filename);  }
  if (useCSV_3D_Output)  { snprintf(csvFilename3D,512,"%s/3d_%s",directory,filename);  }
  if (useCSV_BVH_Output) { snprintf(csvFilenameBVH,512,"%s/bvh_%s",directory,filename);}

  int did2DOutputPreExist  = bvhExportFileExists(csvFilename2D);
  int did3DOutputPreExist  = bvhExportFileExists(csvFilename3D);
  int didBVHOutputPreExist = bvhExportFileExists(csvFilenameBVH);

  //fprintf(stderr,"Wipe 2D:%u 3D:%u BVH:%u\n",wipe_2D_Output,wipe_3D_Output,wipe_BVH_Output);
  if (wipe_2D_Output)  { did2DOutputPreExist=0;   bvhExportFileWipe(csvFilename2D); }
  if (wipe_3D_Output)  { did3DOutputPreExist=0;   bvhExportFileWipe(csvFilename3D); }
  if (wipe_BVH_Output) { didBVHOutputPreExist=0;  bvhExportFileWipe(csvFilenameBVH);}
  //fprintf(stderr,"Pre-exist 2D:%u 3D:%u BVH:%u\n",did2DOutputPreExist,did3DOutputPreExist,didBVHOutputPreExist);

  struct simpleRenderer renderer={0};
  //Declare and populate the simpleRenderer that will project our 3D points

  if (renderConfig->isDefined)
  {
    renderer.fx     = renderConfig->fX;
    renderer.fy     = renderConfig->fY;
    renderer.skew   = 1.0;
    renderer.cx     = renderConfig->cX;
    renderer.cy     = renderConfig->cY;
    renderer.near   = renderConfig->near;
    renderer.far    = renderConfig->far;
    renderer.width  = renderConfig->width;
    renderer.height = renderConfig->height;

    //renderer.cameraOffsetPosition[4];
    //renderer.cameraOffsetRotation[4];
    //renderer.removeObjectPosition;


    //renderer.projectionMatrix[16];
    //renderer.viewMatrix[16];
    //renderer.modelMatrix[16];
    //renderer.modelViewMatrix[16];
    //renderer.viewport[4];

    simpleRendererInitializeFromExplicitConfiguration(&renderer);
    //bvh_freeTransform(&bvhTransform);


    fprintf(stderr,"Direct Rendering is not implemented yet, please don't use it..\n");
    exit(1);
  } else
  {
   //This is the normal rendering where we just simulate our camera center
   simpleRendererDefaults(
                          &renderer,
                          renderConfig->width,
                          renderConfig->height,
                          renderConfig->fX,
                          renderConfig->fY
                         );
    simpleRendererInitialize(&renderer);
  }



  //If we are dumping to CSV  we need to populate the header, this happens only one time for the file
  //------------------------------------------------------------------------------------------
  if (convertToCSV)
   {
    dumpBVHToCSVHeader(
                        mc,
                        csvFilename2D,
                        csvFilename3D,
                        csvFilenameBVH
                      );
   }
  //------------------------------------------------------------------------------------------

  //JSON output
  //------------------------------------------------------------------------------------------
   if (convertToJSON)
   {
    dumpBVHToJSONHeader(
                        mc,
                        wipe_2D_Output,
                        wipe_3D_Output,
                        wipe_BVH_Output,
                        &did2DOutputPreExist,
                        &did3DOutputPreExist,
                        &didBVHOutputPreExist,
                        csvFilename2D,
                        csvFilename3D,
                        csvFilenameBVH,
                        renderer.fx,
                        renderer.fy,
                        renderer.cx,
                        renderer.cy,
                        renderer.near,
                        renderer.far,
                        renderer.width,
                        renderer.height
                       );
   }
  //------------------------------------------------------------------------------------------








  //------------------------------------------------------------------------------------------
  //                                    For every frame
  //------------------------------------------------------------------------------------------
  unsigned int framesDumped=0;
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   if ( (sampleSkip==0) || (fID%sampleSkip==0) )
   {
    if (
        performPointProjectionsForFrame(
                                        mc,
                                        &bvhTransform,
                                        fID,
                                        &renderer,
                                        occlusions,
                                        renderConfig->isDefined
                                       )
       )
    {
     if (
         bvhExportSkeletonFilter(
                                mc,
                                &bvhTransform,
                                &renderer,
                                filterStats,
                                filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                                filterOutSkeletonsWithAnyLimbsOutOfImage,
                                filterWeirdSkeletons
                               )
        )
    {

  //Having projected our BVH data to 3D points using our simpleRenderer Configuration we can store our output to CSV or SVG files..

  //CSV output
  //------------------------------------------------------------------------------------------
      if (convertToCSV)
      {
            dumpBVHToCSVBody(
                             mc,
                             &bvhTransform,
                             &renderer,
                             fID,
                             csvFilename2D,
                             csvFilename3D,
                             csvFilenameBVH,
                             filterStats,
                             filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                             filterOutSkeletonsWithAnyLimbsOutOfImage,
                             filterWeirdSkeletons,
                             encodeRotationsAsRadians
                            );
      }
      //------------------------------------------------------------------------------------------




      //JSON output
      //------------------------------------------------------------------------------------------
      if (convertToJSON)
      {
            //fprintf(stderr,"Pre-exist out 2D:%u 3D:%u BVH:%u\n",did2DOutputPreExist,did3DOutputPreExist,didBVHOutputPreExist);
            dumpBVHToJSONBody(
                              mc,
                              &bvhTransform,
                              &renderer,
                              fID,
                              csvFilename2D,
                              csvFilename3D,
                              csvFilenameBVH,
                              did2DOutputPreExist,
                              did3DOutputPreExist,
                              didBVHOutputPreExist,
                              filterStats,
                              filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                              filterOutSkeletonsWithAnyLimbsOutOfImage,
                              filterWeirdSkeletons,
                              encodeRotationsAsRadians
                             );
      }
     //------------------------------------------------------------------------------------------



     //SVG output
     //------------------------------------------------------------------------------------------
     if (convertToSVG)
     {
      snprintf(svgFilename,512,"%s/%06u.svg",directory,fID);
      framesDumped +=  dumpBVHToSVGFrame(
                                        svgFilename,
                                        mc,
                                        &bvhTransform,
                                        fID,
                                        &renderer
                                       );
      }

     did2DOutputPreExist=1;
     did3DOutputPreExist=1;
     didBVHOutputPreExist=1;
   }//Skeleton is ok to be dumped
  } else //3D Projection was ok
  { fprintf(stderr,RED "Could not perform projection for frame %u\n" NORMAL,fID); }
   //wipe_2D_Output=0;
   //wipe_3D_Output=0;
   //wipe_BVH_Output=0;


   }// Sample skip control..!
  //------------------------------------------------------------------------------------------
  } //For every frame.. ----------------------------------------------------------------------
  //------------------------------------------------------------------------------------------

  //JSON output
  //------------------------------------------------------------------------------------------
  /*
   if (convertToJSON)
   {
    dumpBVHToJSONFooter(
                        mc,
                        csvFilename2D,
                        csvFilenameBVH
                       );
   }*/
  //------------------------------------------------------------------------------------------



  //------------------------------------------------------------------------------------------
  //                            GIVE FINAL CONSOLE OUTPUT HERE
  //------------------------------------------------------------------------------------------
  if (filterStats->visibleJoints!=0)
  {
    fprintf(stderr,"Joint Visibility = %0.2f %%\n",(float) 100*filterStats->invisibleJoints/filterStats->visibleJoints);
  }
  fprintf(stderr,"CSV Outputs: 2D:%d, 3D:%d, BVH:%d\n",useCSV_2D_Output,useCSV_3D_Output,useCSV_BVH_Output);
  //------------------------------------------------------------------------------------------
  fprintf(stderr,"Joints : %u invisible / %u visible ",filterStats->invisibleJoints,filterStats->visibleJoints);
  if (occlusions) { fprintf(stderr,"(occlusions enabled)\n"); } else
                  { fprintf(stderr,"(occlusions disabled)\n");}
  if (renderConfig->isDefined)  { fprintf(stderr,"External Renderer Configuration\n");  } else
                                { fprintf(stderr,"Regular camera position rendering\n");}
  fprintf(stderr,"Filtered out CSV poses : %u\n",filterStats->filteredOutCSVPoses);
  fprintf(stderr,"Filtered behind camera : %u\n",filterStats->filteredOutCSVBehindPoses);
  fprintf(stderr,"Filtered out of camera frame : %u\n",filterStats->filteredOutCSVOutPoses);
  if (mc->numberOfFrames!=0)
  {
   float usedPercentage = (float) 100*(mc->numberOfFrames-filterStats->filteredOutCSVPoses)/mc->numberOfFrames;
   if (usedPercentage==0.0) { fprintf(stderr,RED "----------------\n----------------\n----------------\n"); }
   if (sampleSkip!=0)
   { fprintf(stderr,YELLOW "Sample-skip setting was set to 1 sample for every %u\n" NORMAL,sampleSkip); }
   fprintf(stderr,"Used %0.2f%% of dataset\n",usedPercentage);
   if (usedPercentage==0.0) { fprintf(stderr, "----------------\n----------------\n----------------\n" NORMAL); }

  }
  //------------------------------------------------------------------------------------------
  //------------------------------------------------------------------------------------------

 bvh_freeTransform(&bvhTransform);

 return (framesDumped==mc->numberOfFrames);
}
