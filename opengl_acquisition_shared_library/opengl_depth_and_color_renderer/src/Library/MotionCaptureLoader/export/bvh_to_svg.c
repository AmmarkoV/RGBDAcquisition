#include <stdio.h>
#include <stdlib.h>
#include "bvh_to_svg.h"

#include "../calculate/bvh_project.h"

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


   FILE * fp = fopen(svgFilename,"w");
   if (fp!=0)
   {
      fprintf(fp,"<svg width=\"%u\" height=\"%u\">\n",width,height);
      fprintf(fp,"<rect width=\"%u\" height=\"%u\" style=\"fill:rgb(100,100,100);stroke-width:3;stroke:rgb(100,100,100)\" />\n",width,height);
      fprintf(fp,"<text x=\"10\" y=\"15\">Frame %u</text>\n",fID);
      fprintf(fp,"<text x=\"10\" y=\"30\">Model Position %0.2f,%0.2f,%0.2f</text>\n",
              renderer->cameraOffsetPosition[0]*10,renderer->cameraOffsetPosition[1]*10,renderer->cameraOffsetPosition[2]*10);

      unsigned int x=10;
      float * m = renderer->projectionMatrix.m;
      fprintf(fp,"<text x=\"%u\" y=\"45\">Projection Matrix</text>\n",x);
      fprintf(fp,"<text x=\"%u\" y=\"60\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[0],m[1],m[2],m[3]);
      fprintf(fp,"<text x=\"%u\" y=\"75\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[4],m[5],m[6],m[7]);
      fprintf(fp,"<text x=\"%u\" y=\"90\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[8],m[9],m[10],m[11]);
      fprintf(fp,"<text x=\"%u\" y=\"105\">%0.2f %0.2f %0.2f %0.2f</text>\n",x,m[12],m[13],m[14],m[15]);


      x=150;
      m = renderer->modelViewMatrix.m;
      fprintf(fp,"<text x=\"%u\" y=\"45\">ModelView Matrix</text>\n",x);
      fprintf(fp,"<text x=\"%u\" y=\"60\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[0],m[1],m[2],m[3]);
      fprintf(fp,"<text x=\"%u\" y=\"75\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[4],m[5],m[6],m[7]);
      fprintf(fp,"<text x=\"%u\" y=\"90\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[8],m[9],m[10],m[11]);
      fprintf(fp,"<text x=\"%u\" y=\"105\">%0.2f %0.2f %0.2f %0.2f</text>\n",x,m[12],m[13],m[14],m[15]);

      x=300;
      m = renderer->viewMatrix.m;
      fprintf(fp,"<text x=\"%u\" y=\"45\">View Matrix</text>\n",x);
      fprintf(fp,"<text x=\"%u\" y=\"60\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[0],m[1],m[2],m[3]);
      fprintf(fp,"<text x=\"%u\" y=\"75\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[4],m[5],m[6],m[7]);
      fprintf(fp,"<text x=\"%u\" y=\"90\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[8],m[9],m[10],m[11]);
      fprintf(fp,"<text x=\"%u\" y=\"105\">%0.2f %0.2f %0.2f %0.2f</text>\n",x,m[12],m[13],m[14],m[15]);

      x=450;
      m = renderer->modelMatrix.m;
      fprintf(fp,"<text x=\"%u\" y=\"45\">Model Matrix</text>\n",x);
      fprintf(fp,"<text x=\"%u\" y=\"60\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[0],m[1],m[2],m[3]);
      fprintf(fp,"<text x=\"%u\" y=\"75\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[4],m[5],m[6],m[7]);
      fprintf(fp,"<text x=\"%u\" y=\"90\">%0.2f %0.2f %0.2f %0.2f</text>\n" ,x,m[8],m[9],m[10],m[11]);
      fprintf(fp,"<text x=\"%u\" y=\"105\">%0.2f %0.2f %0.2f %0.2f</text>\n",x,m[12],m[13],m[14],m[15]);

      unsigned int jID=0;
      for (jID=0; jID<mc->jointHierarchySize; jID++)
      {
        unsigned int parentJID = mc->jointHierarchy[jID].parentJoint;

        fprintf(
                fp,"<line x1=\"%0.2f\" y1=\"%0.2f\" x2=\"%0.2f\" y2=\"%0.2f\" style=\"stroke:rgb(255,255,0);stroke-width:2;stroke-dasharray:10,10\" />\n",
                bvhTransform->joint[parentJID].pos2D[0],
                bvhTransform->joint[parentJID].pos2D[1],
                bvhTransform->joint[jID].pos2D[0],
                bvhTransform->joint[jID].pos2D[1]
               );
      }


      //TORSO highlight
      fprintf(
              fp,
              "<rect x=\"%0.2f\" y=\"%0.2f\" width=\"%0.2f\" height=\"%0.2f\" style=\"fill:blue;stroke:pink;stroke-width:5;fill-opacity:0.1;stroke-opacity:0.9\" />\n",
              bvhTransform->torso.rectangle2D.x,
              bvhTransform->torso.rectangle2D.y,
              bvhTransform->torso.rectangle2D.width,
              bvhTransform->torso.rectangle2D.height
             );





      unsigned int occludedListY=130;
      for (jID=0; jID<mc->jointHierarchySize; jID++)
      {

        if (bvhTransform->joint[jID].isOccluded)
        {
           unsigned int parentJID = mc->jointHierarchy[jID].parentJoint;
           occludedListY+=15;
           fprintf(
                   fp,
                   "<text x=\"50\" y=\"%u\">Occluded %s/%s</text>\n",
                   occludedListY,
                   mc->jointHierarchy[parentJID].jointName,
                   mc->jointHierarchy[jID].jointName
                   );

        } else
        {
          fprintf(
                  fp,"<circle cx=\"%0.2f\" cy=\"%0.2f\" r=\"3.00\" stroke=\"rgb(135,135,0)\" stroke-width=\"3\" fill=\"rgb(255,255,0)\" />\n",
                  bvhTransform->joint[jID].pos2D[0],
                  bvhTransform->joint[jID].pos2D[1]
                 );
        }
/*
        fprintf(
                fp,"<text x=\"%0.2f\" y=\"%0.2f\">%0.2f,%0.2f,%0.2f</text>\n",
                bvhTransform->joint[jID].pos2D[0]+10,
                bvhTransform->joint[jID].pos2D[1],
                bvhTransform->joint[jID].pos3D[0],
                bvhTransform->joint[jID].pos3D[1],
                bvhTransform->joint[jID].pos3D[2]
               );
*/
        fprintf(
                fp,"<!-- x=\"%0.2f\" y=\"%0.2f\" z=\"%0.2f\" -->\n",
                bvhTransform->joint[jID].pos3D[0],
                bvhTransform->joint[jID].pos3D[1],
                bvhTransform->joint[jID].pos3D[2]
               );
      }


      fprintf(fp,"</svg>\n");
      fclose(fp);
     return 1;
   }
 return 0;
}
