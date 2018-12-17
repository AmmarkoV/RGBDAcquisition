#include <stdio.h>
#include <stdlib.h>
#include "bvh_export.h"
#include "bvh_to_csv.h"
#include "bvh_to_svg.h"

#include "../bvh_project.h"


int performPointProjections(
                             struct BVH_MotionCapture * mc,
                             struct BVH_Transform * bvhTransform,
                             unsigned int fID,
                             struct simpleRenderer * renderer,
                             unsigned int occlusions
                            )
{
  //First load the 3D positions of each joint..
  if (
       bvh_loadTransformForFrame(
                                 mc,
                                 fID ,
                                 bvhTransform
                                )
      )
       {
        //Then project 3D positions on 2D frame and save results..
        if (
            bvh_projectTo2D(
                            mc,
                            bvhTransform,
                            renderer,
                            occlusions
                           )
           )
           {
             return 1;
           }
       }  else
       {
           bvh_cleanTransform(
                              mc,
                              bvhTransform
                             );
       }
   //----------------------------------------------------------
 return 0;
}




int dumpBVHToSVGCSV(
                    const char * directory,
                    const char * filename,
                    int convertToSVG,
                    int convertToCSV,
                    struct BVH_MotionCapture * mc,
                    unsigned int width,
                    unsigned int height,
                    unsigned int occlusions,
                    unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                    unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                    unsigned int filterWeirdSkeletons,
                    unsigned int encodeRotationsAsRadians
                   )
{
  struct BVH_Transform bvhTransform;
  char svgFilename[512];
  char csvFilename[512];


  struct simpleRenderer renderer={0};
  renderer.width=width;
  renderer.height=height;
  renderer.fx = 575.816;
  renderer.fy = 575.816;
  renderer.skew = 0.0;
  renderer.cx = (float) width/2;
  renderer.cy = (float) height/2;
  renderer.near = 1.0;
  renderer.far = 1000.0;

  //-----------------------------------
  renderer.cameraOffsetPosition[0]=0.0;
  renderer.cameraOffsetPosition[1]=0.0;
  renderer.cameraOffsetPosition[2]=0.0;
  renderer.cameraOffsetPosition[3]=0.0;
  //-----------------------------------
  renderer.cameraOffsetRotation[0]=0.0;
  renderer.cameraOffsetRotation[1]=0.0;
  renderer.cameraOffsetRotation[2]=0.0;
  renderer.cameraOffsetRotation[3]=0.0;
  //-----------------------------------

  simpleRendererInitialize(&renderer);


  snprintf(csvFilename,512,"%s/%s",directory,filename);
  if (convertToCSV)
   {
    dumpBVHToCSVHeader(
                        mc,
                        csvFilename
                       );
   }


  unsigned int framesDumped=0;
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   snprintf(svgFilename,512,"%s/%06u.svg",directory,fID);
   if (
       !performPointProjections(
                                mc,
                                &bvhTransform,
                                fID,
                                &renderer,
                                occlusions
                               )
       )
   {
       fprintf(stderr,"Could not perform projection for frame %u\n",fID);
   }

   if (convertToCSV)
   {
      dumpBVHToCSVBody(
                       mc,
                       &bvhTransform,
                       &renderer,
                       fID,
                       csvFilename,
                       filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                       filterOutSkeletonsWithAnyLimbsOutOfImage,
                       filterWeirdSkeletons,
                       encodeRotationsAsRadians
                      );
   }


   if (convertToSVG)
   {
    framesDumped +=  dumpBVHToSVGFrame(
                                       svgFilename,
                                       mc,
                                       &bvhTransform,
                                       fID,
                                       &renderer
                                      );
   }

  }

  fprintf(stderr,"Filtered out CSV poses : %u\n",filteredOutCSVPoses);
  fprintf(stderr,"Filtered behind camera : %u\n",filteredOutCSVBehindPoses);
  fprintf(stderr,"Filtered out of camera frame : %u\n",filteredOutCSVOutPoses);
  if (mc->numberOfFrames!=0)
  {
   fprintf(stderr,"Used %0.2f%% of dataset\n",(float) 100*(mc->numberOfFrames-filteredOutCSVPoses)/mc->numberOfFrames);
  }


 return (framesDumped==mc->numberOfFrames);
}
