#include <stdio.h>
#include <stdlib.h>
#include "bvh_to_svg.h"

#include "../bvh_project.h"

int dumpBVHToSVGFrame(
                      const char * svgFilename,
                      struct BVH_MotionCapture * mc,
                      struct BVH_Transform * bvhTransform,
                      unsigned int fID,
                      struct simpleRenderer * renderer
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
