#include <stdio.h>
#include "bvh_to_svg.h"
#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../../tools/AmMatrix/matrixOpenGL.h"
#include "../../../../../tools/AmMatrix/simpleRenderer.h"



int dumpBVHToSVGFile(
                     const char * filename,
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform * bvhTransform,
                     unsigned int fID,
                     struct simpleRenderer * renderer,
                     float * positionOffset
                    )
{
   unsigned int width = renderer->width;
   unsigned int height = renderer->height;

   float position2DX;
   float position2DY;

   FILE * fp = fopen(filename,"w");
   if (fp!=0)
   {
      fprintf(fp,"<svg width=\"%u\" height=\"%u\">\n",width,height);
      fprintf(fp,"<rect width=\"%u\" height=\"%u\" style=\"fill:rgb(255,255,255);stroke-width:3;stroke:rgb(255,255,255)\" />\n",width,height);
      fprintf(fp,"<text x=\"10\" y=\"40\">Frame %u</text>\n",fID);


      //First load the 3D positions of each joint..
      bvh_loadTransformForFrame(
                                mc,
                                fID ,
                                bvhTransform
                               );
      //Then project 3D positions on 2D frame and save results..
      unsigned int jID=0;
      for (jID=0; jID<mc->jointHierarchySize; jID++)
      {

        if (bvhTransform->joint[jID].pos3D[3]!=1.0)
        {
          fprintf(stderr,"bvh_loadTransformForFrame location for joint %u not normalized..\n",jID);
        }

           float pos3DFloat[4];
           pos3DFloat[0]=(float)bvhTransform->joint[jID].pos3D[0]+positionOffset[0];
           pos3DFloat[1]=(float)bvhTransform->joint[jID].pos3D[1]+positionOffset[1];
           pos3DFloat[2]=(float)bvhTransform->joint[jID].pos3D[2]+positionOffset[2];
           pos3DFloat[3]=0.0;
           simpleRendererRender(
                                 renderer ,
                                 pos3DFloat,
                                 &position2DX,
                                 &position2DY
                                );


        fprintf(fp,"<circle cx=\"%0.2f\" cy=\"%0.2f\" r=\"8.00\" stroke=\"rgb(135,135,0)\" stroke-width=\"3\" fill=\"rgb(255,255,0)\" />\n",
                  position2DX,
                  position2DY);
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
                 float * positionOffset
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


  simpleRendererInitialize(&renderer);

  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   snprintf(filename,512,"%s/%06u.svg",directory,fID);

   framesDumped +=  dumpBVHToSVGFile(
                                     filename,
                                     mc,
                                     &bvhTransform,
                                     fID,
                                     &renderer,
                                      positionOffset
                                     );
  }

 return (framesDumped==mc->numberOfFrames);
}
