#include <stdio.h>
#include "bvh_to_svg.h"




int dumpBVHToSVGFile(
                     const char * filename,
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform * bvhTransform,
                     unsigned int fID,
                     unsigned int width,
                     unsigned int height
                    )
{
   FILE * fp = fopen(filename,"w");
   if (fp!=0)
   {
      fprintf(fp,"<svg width=\"%u\" height=\"%u\">\n",width,height);
      fprintf(fp,"<rect width=\"%u\" height=\"%u\" style=\"fill:rgb(255,255,255);stroke-width:3;stroke:rgb(255,255,255)\" />\n",width,height);
      fprintf(fp,"<text x=\"10\" y=\"40\">Frame %u</text>\n",fID);

//<circle cx="704.26" cy="505.16" r="8.00" stroke="rgb(135,135,0)" stroke-width="3" fill="rgb(255,255,0)" />




      bvh_loadTransformForFrame(
                                mc,
                                fID ,
                                bvhTransform
                               );



      fprintf(fp,"</svg>\n");
     return 1;
   }
 return 0;
}







int dumpBVHToSVG(
                 const char * directory ,
                 struct BVH_MotionCapture * mc,
                 unsigned int width,
                 unsigned int height
                 )
{
  struct BVH_Transform bvhTransform;
  char filename[512];

  unsigned int framesDumped=0;
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   snprintf(filename,512,"%s/%06u.svg",directory,fID);

   framesDumped +=  dumpBVHToSVGFile(
                                     filename,
                                     mc,
                                     &bvhTransform,
                                     fID,
                                     width,
                                     height
                                    );
  }

 return (framesDumped==mc->numberOfFrames);
}
