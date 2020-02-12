#include "bvh_filter.h"


#include "../export/bvh_export.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int filterOutPosesThatAreCloseToRules(struct BVH_MotionCapture * mc,int argc,const char **argv)
{
 for (int i=0; i<argc; i++)
 {
   fprintf(stderr,"%02u| %s \n",i,argv[i]);
 }

 if (argc<5)
 {
   fprintf(stderr,"filterOutPosesThatAreCloseToRules: Too few arguments..\n");
   return 0;
 }

 unsigned int width=atoi(argv[0]);
 unsigned int height=atoi(argv[1]);
 float fX=atof(argv[2]);
 float fY=atof(argv[3]);

 unsigned int numberOfRules=atoi(argv[4]);

 if (argc<5+numberOfRules*4)
 {
   fprintf(stderr,"filterOutPosesThatAreCloseToRules: Too few arguments at rules..\n");
   return 0;
 }




  struct simpleRenderer renderer={0};
  //Declare and populate the simpleRenderer that will project our 3D points

   //This is the normal rendering where we just simulate our camera center
   simpleRendererDefaults(
                          &renderer,
                          width,
                          height,
                          fX,
                          fY
                         );
   simpleRendererInitialize(&renderer);




  struct BVH_Transform bvhTransform;
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   if (
        performPointProjectionsForFrame(
                                        mc,
                                        &bvhTransform,
                                        fID,
                                        &renderer,
                                        0,//Occlusions
                                        0 //Direct Rendering
                                       )
       )
   {
    for (int rule=0; rule<numberOfRules; rule++)
     {
      fprintf(stderr,"filterOutPosesThatAreCloseToRules rule #%u\n",rule);
      const char * jointA = argv[5+rule*4+0];
      const char * jointB = argv[5+rule*4+1];
      float  minimumDistance = atof(argv[5+rule*4+2]);
      float  maxiumDistance = atof(argv[5+rule*4+3]);
      fprintf(stderr,"%s -> %s =|> min(%0.2f) max(%0.2f)\n",jointA,jointB,minimumDistance,maxiumDistance);
     }
  }
 }






 return 1;
}
