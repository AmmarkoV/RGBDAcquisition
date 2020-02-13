#include "bvh_filter.h"


#include "../export/bvh_export.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


float getDistanceBetweenJoints(struct BVH_Transform * bvhTransform,BVHJointID * jIDA,BVHJointID * jIDB)
{
  float xA = bvhTransform->joint[*jIDA].pos2D[0];
  float yA = bvhTransform->joint[*jIDA].pos2D[1];
  //--------------------------------------------
  float xB = bvhTransform->joint[*jIDB].pos2D[0];
  float yB = bvhTransform->joint[*jIDB].pos2D[1];

  return sqrt( (xA-xB)*(xA-xB) + (yA-yB)*(yA-yB) );
}


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



   float forcePosition[3]={0.0,0.0,-130.0};
   float forceRotation[3]={0.0,0.0,0.0};

  struct BVH_Transform bvhTransform;
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   if (
           performPointProjectionsForFrameForcingPositionAndRotation(
                                                                     mc,
                                                                     &bvhTransform,
                                                                     fID,
                                                                     &renderer,
                                                                     forcePosition,
                                                                     forceRotation,
                                                                     0,//Occlusions
                                                                     0//Direct Rendering
                                                                    )
       )
   {
    int rulesThatApplyForFrame=0;
    for (int rule=0; rule<numberOfRules; rule++)
     {
      //fprintf(stderr,"filterOutPosesThatAreCloseToRules rule #%u\n",rule);
      const char * jointA = argv[5+rule*4+0];
      BVHJointID jIDA=0;
      const char * jointB = argv[5+rule*4+1];
      BVHJointID jIDB=0;
      float  minimumDistance = atof(argv[5+rule*4+2]);
      float  maximumDistance = atof(argv[5+rule*4+3]);

      if (
          (
           bvh_getJointIDFromJointNameNocase(
                                             mc,
                                             jointA,
                                             &jIDA
                                            )
          )
           &&
          (
           bvh_getJointIDFromJointNameNocase(
                                             mc,
                                             jointB,
                                             &jIDB
                                            )
          )
         )
         {
           fprintf(stderr,"%s -> %s =|> min(%0.2f) max(%0.2f)\n",jointA,jointB,minimumDistance,maximumDistance);

           float value = getDistanceBetweenJoints(&bvhTransform,&jIDA,&jIDB);

           if (
                (minimumDistance<value) &&
                (value<maximumDistance)
                )
               {
                 ++rulesThatApplyForFrame;
               }
         }
     }

     if (rulesThatApplyForFrame==numberOfRules)
     {
       //Frame matches our rules so we must hide it..!
       fprintf(stderr,"Frame %u Match\n",fID);
     }

  }
 }


 return 1;
}







int probeForFilterRules(struct BVH_MotionCapture * mc,int argc,const char **argv)
{
 for (int i=0; i<argc; i++)
 {
   fprintf(stderr,"%02u| %s \n",i,argv[i]);
 }

 if (argc<5)
 {
   fprintf(stderr,"probeForFilterRules: Too few arguments..\n");
   return 0;
 }

 unsigned int width=atoi(argv[0]);
 unsigned int height=atoi(argv[1]);
 float fX=atof(argv[2]);
 float fY=atof(argv[3]);

 unsigned int numberOfRules=atoi(argv[4]);

 if (argc<5+numberOfRules*4)
 {
   fprintf(stderr,"probeForFilterRules: Too few arguments at rules..\n");
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


   float forcePosition[3]={0.0,0.0,-130.0};
   float forceRotation[3]={0.0,0.0,0.0};



  struct BVH_Transform bvhTransform;
  unsigned int fID=0;
  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   if (
           performPointProjectionsForFrameForcingPositionAndRotation(
                                                                     mc,
                                                                     &bvhTransform,
                                                                     fID,
                                                                     &renderer,
                                                                     forcePosition,
                                                                     forceRotation,
                                                                     0,//Occlusions
                                                                     0//Direct Rendering
                                                                    )
       )
   {
    int rulesThatApplyForFrame=0;
    for (int rule=0; rule<numberOfRules; rule++)
     {
      //fprintf(stderr,"filterOutPosesThatAreCloseToRules rule #%u\n",rule);
      const char * jointA = argv[5+rule*4+0];
      BVHJointID jIDA=0;
      const char * jointB = argv[5+rule*4+1];
      BVHJointID jIDB=0;
      float  minimumDistance = atof(argv[5+rule*4+2]);
      float  maximumDistance = atof(argv[5+rule*4+3]);

      if (
          (
           bvh_getJointIDFromJointNameNocase(
                                             mc,
                                             jointA,
                                             &jIDA
                                            )
          )
           &&
          (
           bvh_getJointIDFromJointNameNocase(
                                             mc,
                                             jointB,
                                             &jIDB
                                            )
          )
         )
         {
           float value = getDistanceBetweenJoints(&bvhTransform,&jIDA,&jIDB);
           fprintf(stderr,"Frame %u -> Distance(%s,%s) = %0.2f pixels\n",fID,jointA,jointB,value);
         }
     } //Apply each rule
  } //Transform correctly calculated..
 } //End of loop for every BVH frame


 return 1;
}

