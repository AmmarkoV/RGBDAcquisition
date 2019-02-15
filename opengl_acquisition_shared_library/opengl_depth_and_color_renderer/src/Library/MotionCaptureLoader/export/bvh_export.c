#include <stdio.h>
#include <stdlib.h>
#include "bvh_export.h"
#include "bvh_to_csv.h"
#include "bvh_to_svg.h"



int actuallyPerformPointProjections(
                                    struct BVH_MotionCapture * mc,
                                    struct BVH_Transform * bvhTransform,
                                    struct simpleRenderer * renderer,
                                    unsigned int occlusions
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

   //----------------------------------------------------------
 return 0;
}


int performPointProjectionsForFrame(
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
         return actuallyPerformPointProjections(mc,bvhTransform,renderer,occlusions);
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



int performPointProjectionsForMotionBuffer(
                                            struct BVH_MotionCapture * mc,
                                            struct BVH_Transform * bvhTransform,
                                            float * motionBuffer,
                                            struct simpleRenderer * renderer,
                                            unsigned int occlusions
                                           )
{
  //First load the 3D positions of each joint..
  if (
       bvh_loadTransformForMotionBuffer(
                                        mc ,
                                        motionBuffer,
                                        bvhTransform
                                       )
      )
       {
        //Then project 3D positions on 2D frame and save results..
         return actuallyPerformPointProjections(mc,bvhTransform,renderer,occlusions);
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
                    float fX,
                    float fY,
                    unsigned int occlusions,
                    unsigned int filterOutSkeletonsWithAnyLimbsBehindTheCamera,
                    unsigned int filterOutSkeletonsWithAnyLimbsOutOfImage,
                    unsigned int filterWeirdSkeletons,
                    unsigned int encodeRotationsAsRadians
                   )
{
  struct BVH_Transform bvhTransform;
  char svgFilename[512];
  char csvFilename2D[512];
  char csvFilename3D[512];
  char csvFilenameBVH[512];



  struct simpleRenderer renderer={0};

  simpleRendererDefaults(&renderer,width,height,fX,fY);
  simpleRendererInitialize(&renderer);


  snprintf(csvFilename2D,512,"%s/2d_%s",directory,filename);
  snprintf(csvFilename3D,512,"%s/3d_%s",directory,filename);
  snprintf(csvFilenameBVH,512,"%s/bvh_%s",directory,filename);
  if (convertToCSV)
   {
    dumpBVHToCSVHeader(
                        mc,
                        csvFilename2D,
                        csvFilename3D,
                        csvFilenameBVH
                       );
   }


  unsigned int framesDumped=0;
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   snprintf(svgFilename,512,"%s/%06u.svg",directory,fID);
   if (
       !performPointProjectionsForFrame(
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
                       csvFilename2D,
                       csvFilename3D,
                       csvFilenameBVH,
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

  if (visibleJoints!=0)
  {
    fprintf(stderr,"Joint Visibility = %0.2f %%\n",(float) 100*invisibleJoints/visibleJoints);
  }

  fprintf(stderr,"Joints : %u invisible / %u visible ",invisibleJoints,visibleJoints);
  if (occlusions) { fprintf(stderr,"(occlusions enabled)\n"); } else
                  { fprintf(stderr,"(occlusions disabled)\n");      }
  fprintf(stderr,"Filtered out CSV poses : %u\n",filteredOutCSVPoses);
  fprintf(stderr,"Filtered behind camera : %u\n",filteredOutCSVBehindPoses);
  fprintf(stderr,"Filtered out of camera frame : %u\n",filteredOutCSVOutPoses);
  if (mc->numberOfFrames!=0)
  {
   fprintf(stderr,"Used %0.2f%% of dataset\n",(float) 100*(mc->numberOfFrames-filteredOutCSVPoses)/mc->numberOfFrames);
  }


 return (framesDumped==mc->numberOfFrames);
}
