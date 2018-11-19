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
                             float * objectRotationOffset
                            )
{


  //First load the 3D positions of each joint..
  bvh_loadTransformForFrame(
                            mc,
                            fID ,
                            bvhTransform
                           );
  //Then project 3D positions on 2D frame and save results..
  bvh_projectTo2D(
                  mc,
                  bvhTransform,
                  renderer,
                  objectRotationOffset
                 );
   //----------------------------------------------------------

 return 1;
}




int dumpBVHToSVGCSV(
                    const char * directory,
                    const char * filename,
                    int convertToSVG,
                    int convertToCSV,
                    struct BVH_MotionCapture * mc,
                    unsigned int width,
                    unsigned int height,
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

  BVHJointID rootJID;
  bvh_getRootJointID( mc, &rootJID );



  unsigned int framesDumped=0;
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   snprintf(svgFilename,512,"%s/%06u.svg",directory,fID);


    float objectRotationOffsetCopy[3]={0};


    float data[8]={0};
    if (bhv_populatePosXYZRotXYZ(mc,rootJID,fID,data,sizeof(data)))
         {
            renderer.cameraOffsetPosition[0]=data[0];
            renderer.cameraOffsetPosition[1]=data[1]-80;
            renderer.cameraOffsetPosition[2]=data[2]+300;
            renderer.cameraOffsetPosition[3]=0.0;
            objectRotationOffsetCopy[0]=data[3]+0;
            objectRotationOffsetCopy[1]=data[4];
            objectRotationOffsetCopy[2]=data[5];
            renderer.cameraOffsetRotation[3]=0.0;
         }

   performPointProjections(
                           mc,
                           &bvhTransform,
                           fID,
                           &renderer,
                           objectRotationOffsetCopy
                          );

   if (convertToCSV)
   {
      dumpBVHToCSVBody(
                       mc,
                       &bvhTransform,
                       &renderer,
                       objectRotationOffsetCopy,
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
                                       &renderer,
                                       objectRotationOffsetCopy
                                      );
   }

  }
  fprintf(stderr,"Filtered out CSV poses : %u\n",filteredOutCSVPoses);


 return (framesDumped==mc->numberOfFrames);
}
