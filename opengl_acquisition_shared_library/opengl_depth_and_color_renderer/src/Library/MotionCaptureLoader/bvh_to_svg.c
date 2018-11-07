#include <stdio.h>
#include <stdlib.h>
#include "bvh_to_svg.h"

#include "bvh_project.h"

#include "../../../../../tools/AmMatrix/simpleRenderer.h"

#define DUMP_3D_POSITIONS 0

int fileExists(const char * filename)
{
 FILE *fp = fopen(filename,"r");
 if( fp ) { /* exists */ fclose(fp); return 1; }
 return 0;
}



int dumpBVHToCSVHeader(
                       struct BVH_MotionCapture * mc,
                       struct simpleRenderer * renderer,
                       const char * filename
                      )
{
   if (!fileExists(filename))
   {
     FILE * fp =fopen(filename,"w");

   if (fp!=0)
   {
     fprintf(fp,"positionX,positionY,positionZ,roll,pitch,yaw,");


     unsigned int jID=0;
     //2D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
          fprintf(fp,"2DX_%s,2DY_%s,",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName);
         }
       }

     #if DUMP_3D_POSITIONS
     //3D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           fprintf(fp,"3DX_%s,3DY_%s,3DZ_%s,",mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName,mc->jointHierarchy[jID].jointName);
         }
       }
     #endif // DUMP_3D_POSITIONS


     //3D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           fprintf(fp,"%s_%s,%s_%s,%s_%s,",
                   channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[0]],
                   mc->jointHierarchy[jID].jointName,
                   channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[1]],
                   mc->jointHierarchy[jID].jointName,
                   channelNames[(unsigned int) mc->jointHierarchy[jID].channelType[2]],
                   mc->jointHierarchy[jID].jointName);
         }
       }
     fprintf(fp,"frameID\n");

     fclose(fp);
     return 1;
   }
  } else
  {
   fprintf(stderr,"We don't need to regenerate the CSV header, it already exists\n");
   return 1;
  }
 return 0;
}




int dumpBVHToCSVBody(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform * bvhTransform,
                       struct simpleRenderer * renderer,
                       float * objectRotationOffset,
                       unsigned int fID,
                       const char * filename
                      )
{
   FILE * fp = fopen(filename,"a");
   if (fp!=0)
   {

    fprintf(fp,"%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,%0.2f,",
                 renderer->cameraOffsetPosition[0]*10,
                 renderer->cameraOffsetPosition[1]*10,
                 renderer->cameraOffsetPosition[2]*10,
                 objectRotationOffset[2],
                 objectRotationOffset[1],
                 objectRotationOffset[0]
                 );


     unsigned int jID=0;
     //2D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
          fprintf(
                  fp,"%0.2f,%0.2f,",
                  bvhTransform->joint[jID].pos2D[0],
                  bvhTransform->joint[jID].pos2D[1]
                 );
         }
       }

     #if DUMP_3D_POSITIONS
     //3D Positions
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
          fprintf(
                  fp,"%0.2f,%0.2f,%0.2f,",
                  bvhTransform->joint[jID].pos3D[0],
                  bvhTransform->joint[jID].pos3D[1],
                  bvhTransform->joint[jID].pos3D[2]
                 );
         }
       }
     #endif // DUMP_3D_POSITIONS

     //Joint Configuration
     for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
         if (!mc->jointHierarchy[jID].isEndSite)
         {
           fprintf(
                   fp,"%0.2f,%0.2f,%0.2f,",
                   bvh_getJointChannelAtFrame(mc,jID,fID,(unsigned int) mc->jointHierarchy[jID].channelType[0]),
                   bvh_getJointChannelAtFrame(mc,jID,fID,(unsigned int) mc->jointHierarchy[jID].channelType[1]),
                   bvh_getJointChannelAtFrame(mc,jID,fID,(unsigned int) mc->jointHierarchy[jID].channelType[2])
                   );
         }
       }

     fprintf(fp,"%u\n",fID);

     fclose(fp);
     return 1;
   }
 return 0;
}




int dumpBVHToSVGFrame(
                      const char * svgFilename,
                      struct BVH_MotionCapture * mc,
                      struct BVH_Transform * bvhTransform,
                      unsigned int fID,
                      struct simpleRenderer * renderer,
                      float * objectRotationOffset
                     )
{
   unsigned int width = renderer->width;
   unsigned int height = renderer->height;
   unsigned int jID=0;
   unsigned int parentJID=0;


   FILE * fp = fopen(svgFilename,"w");
   if (fp!=0)
   {
      fprintf(fp,"<svg width=\"%u\" height=\"%u\">\n",width,height);
      fprintf(fp,"<rect width=\"%u\" height=\"%u\" style=\"fill:rgb(100,100,100);stroke-width:3;stroke:rgb(100,100,100)\" />\n",width,height);
      fprintf(fp,"<text x=\"10\" y=\"15\">Frame %u</text>\n",fID);
      fprintf(fp,"<text x=\"10\" y=\"30\">Model Position %0.2f,%0.2f,%0.2f</text>\n",
              renderer->cameraOffsetPosition[0]*10,renderer->cameraOffsetPosition[1]*10,renderer->cameraOffsetPosition[2]*10);
      fprintf(fp,"<text x=\"10\" y=\"45\">Model Euler Rotation %0.2f,%0.2f,%0.2f</text>\n",
              objectRotationOffset[0],objectRotationOffset[1],objectRotationOffset[2]);



      for (jID=0; jID<mc->jointHierarchySize; jID++)
      {
        parentJID = mc->jointHierarchy[jID].parentJoint;

        fprintf(
                fp,"<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" style=\"stroke:rgb(255,255,0);stroke-width:2;stroke-dasharray:10,10\" />\n",
                bvhTransform->joint[parentJID].pos2D[0],
                bvhTransform->joint[parentJID].pos2D[1],
                bvhTransform->joint[jID].pos2D[0],
                bvhTransform->joint[jID].pos2D[1]
               );
      }



      for (jID=0; jID<mc->jointHierarchySize; jID++)
      {
        fprintf(
                fp,"<circle cx=\"%0.2f\" cy=\"%0.2f\" r=\"3.00\" stroke=\"rgb(135,135,0)\" stroke-width=\"3\" fill=\"rgb(255,255,0)\" />\n",
                bvhTransform->joint[jID].pos2D[0],
                bvhTransform->joint[jID].pos2D[1]
               );
      }


      fprintf(fp,"</svg>\n");
      fclose(fp);
     return 1;
   }
 return 0;
}

float randomFloat( float min, float max )
{
    float scale = rand() / (float) RAND_MAX; /* [0, 1.0] */

    float absoluteRandom = scale * (max - min);      /* [min, max] */

    float value = max-absoluteRandom;

    if (value<min) { fprintf(stderr,"randomFloat(%0.2f,%0.2f)=>%0.2f TOO SMALL\n",min,max,value); }
    if (value>max) { fprintf(stderr,"randomFloat(%0.2f,%0.2f)=>%0.2f TOO BIG\n",min,max,value); }

    return value;
}

int performPointProjections(
                             struct BVH_MotionCapture * mc,
                             struct BVH_Transform * bvhTransform,
                             unsigned int fID,
                             struct simpleRenderer * renderer,
                             float * objectRotationOffset,

                             unsigned int randomizePoses,
                             float * minimumObjectPositionValue,
                             float * maximumObjectPositionValue,
                             float * minimumObjectRotationValue,
                             float * maximumObjectRotationValue
                            )
{
  float objectRotationOffsetCopy[3];
  objectRotationOffsetCopy[0]=objectRotationOffset[0];
  objectRotationOffsetCopy[1]=objectRotationOffset[1];
  objectRotationOffsetCopy[2]=objectRotationOffset[2];

  if (randomizePoses)
  {
   renderer->cameraOffsetPosition[0]=randomFloat(minimumObjectPositionValue[0],maximumObjectPositionValue[0]);
   renderer->cameraOffsetPosition[1]=randomFloat(minimumObjectPositionValue[1],maximumObjectPositionValue[1]);
   renderer->cameraOffsetPosition[2]=randomFloat(minimumObjectPositionValue[2],maximumObjectPositionValue[2]);

   objectRotationOffsetCopy[0]=randomFloat(minimumObjectRotationValue[0],maximumObjectRotationValue[0]);
   objectRotationOffsetCopy[1]=randomFloat(minimumObjectRotationValue[1],maximumObjectRotationValue[1]);
   objectRotationOffsetCopy[2]=randomFloat(minimumObjectRotationValue[2],maximumObjectRotationValue[2]);
  }

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
                  objectRotationOffsetCopy
                 );
   //----------------------------------------------------------

 return 1;
}



int dumpBVHToSVG(
                 const char * directory,
                 int convertToSVG,
                 int convertToCSV,
                 struct BVH_MotionCapture * mc,
                 unsigned int width,
                 unsigned int height,
                 float * cameraPositionOffset,
                 float * cameraRotationOffset,
                 float * objectRotationOffset,

                 unsigned int randomizePoses,
                 float * minimumObjectPositionValue,
                 float * maximumObjectPositionValue,
                 float * minimumObjectRotationValue,
                 float * maximumObjectRotationValue
                 )
{
  struct BVH_Transform bvhTransform;
  char svgFilename[512];
  char csvFilename[512];

  unsigned int framesDumped=0;
  unsigned int fID=0;


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

  renderer.cameraOffsetPosition[0]=cameraPositionOffset[0];
  renderer.cameraOffsetPosition[1]=cameraPositionOffset[1];
  renderer.cameraOffsetPosition[2]=cameraPositionOffset[2];
  renderer.cameraOffsetPosition[3]=0.0;

  renderer.removeObjectPosition=1;

  renderer.cameraOffsetRotation[0]=cameraRotationOffset[0];
  renderer.cameraOffsetRotation[1]=cameraRotationOffset[1];
  renderer.cameraOffsetRotation[2]=cameraRotationOffset[2];
  renderer.cameraOffsetRotation[3]=0.0;

  simpleRendererInitialize(&renderer);


  snprintf(csvFilename,512,"%s/data.csv",directory);
  if (convertToCSV)
   {
    dumpBVHToCSVHeader(
                        mc,
                        &renderer,
                        csvFilename
                       );
   }

  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   snprintf(svgFilename,512,"%s/%06u.svg",directory,fID);

   performPointProjections(
                           mc,
                           &bvhTransform,
                           fID,
                           &renderer,
                           objectRotationOffset,

                           randomizePoses,
                           minimumObjectPositionValue,
                           maximumObjectPositionValue,
                           minimumObjectRotationValue,
                           maximumObjectRotationValue
                          );

   if (convertToCSV)
   {
      dumpBVHToCSVBody(
                       mc,
                       &bvhTransform,
                       &renderer,
                       objectRotationOffset,
                       fID,
                       csvFilename
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
                                       objectRotationOffset
                                      );
   }

  }

 return (framesDumped==mc->numberOfFrames);
}
