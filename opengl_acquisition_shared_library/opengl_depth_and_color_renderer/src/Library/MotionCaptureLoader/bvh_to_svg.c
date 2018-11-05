#include <stdio.h>
#include "bvh_to_svg.h"

#include "bvh_project.h"

#include "../../../../../tools/AmMatrix/simpleRenderer.h"

int dumpBVHToSVGFrame(
                     const char * filename,
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


   FILE * fp = fopen(filename,"w");
   if (fp!=0)
   {
      fprintf(fp,"<svg width=\"%u\" height=\"%u\">\n",width,height);
      fprintf(fp,"<rect width=\"%u\" height=\"%u\" style=\"fill:rgb(100,100,100);stroke-width:3;stroke:rgb(100,100,100)\" />\n",width,height);
      fprintf(fp,"<text x=\"10\" y=\"15\">Frame %u</text>\n",fID);
      fprintf(fp,"<text x=\"10\" y=\"30\">Model Position %0.2f,%0.2f,%0.2f</text>\n",
              renderer->cameraOffsetPosition[0]*10,renderer->cameraOffsetPosition[1]*10,renderer->cameraOffsetPosition[2]*10);
      fprintf(fp,"<text x=\"10\" y=\"45\">Model Euler Rotation %0.2f,%0.2f,%0.2f</text>\n",
              objectRotationOffset[0],objectRotationOffset[1],objectRotationOffset[2]);

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
     return 1;
   }
 return 0;
}







int dumpBVHToSVG(
                 const char * directory ,
                 struct BVH_MotionCapture * mc,
                 unsigned int width,
                 unsigned int height,
                 float * cameraPositionOffset,
                 float * cameraRotationOffset,
                 float * objectRotationOffset
                 )
{
  struct BVH_Transform bvhTransform;
  char filename[512];

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

  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   snprintf(filename,512,"%s/%06u.svg",directory,fID);

   framesDumped +=  dumpBVHToSVGFrame(
                                       filename,
                                       mc,
                                       &bvhTransform,
                                       fID,
                                       &renderer,
                                       objectRotationOffset
                                      );
  }

 return (framesDumped==mc->numberOfFrames);
}
