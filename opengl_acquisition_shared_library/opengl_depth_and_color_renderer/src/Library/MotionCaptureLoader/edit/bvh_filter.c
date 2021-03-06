#include "bvh_filter.h"


#include "../export/bvh_export.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>


#define NORMAL   "\033[0m"
#define BLACK   "\033[30m"      /* Black */
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */
#define YELLOW  "\033[33m"      /* Yellow */
#define BLUE    "\033[34m"      /* Blue */
#define MAGENTA "\033[35m"      /* Magenta */
#define CYAN    "\033[36m"      /* Cyan */
#define WHITE   "\033[37m"      /* White */


float getDistanceBetweenJoints(struct BVH_Transform * bvhTransform,BVHJointID * jIDA,BVHJointID * jIDB)
{
  float xA = bvhTransform->joint[*jIDA].pos2D[0];
  float yA = bvhTransform->joint[*jIDA].pos2D[1];
  //--------------------------------------------
  float xB = bvhTransform->joint[*jIDB].pos2D[0];
  float yB = bvhTransform->joint[*jIDB].pos2D[1];

  return sqrt( (xA-xB)*(xA-xB) + (yA-yB)*(yA-yB) );
}






int filterOutPosesThatAreGimbalLocked(struct BVH_MotionCapture * mc,float threshold)
{
  unsigned int * framesToRemove = (unsigned int * ) malloc(sizeof(unsigned int) * mc->numberOfFrames);

  if (framesToRemove==0)
  {
    fprintf(stderr,"filterOutPosesThatAreGimbalLocked: Could not allocate space to accomodate poses..\n");
    return 0;
  }

  if (mc->numberOfValuesPerFrame==0)
  {
    fprintf(stderr,RED "filterOutPosesThatAreGimbalLocked: No motion values detected, have you loaded a file?\n" NORMAL);
    free(framesToRemove);
    return 0;
  }

  memset(framesToRemove,0,sizeof(unsigned int) * mc->numberOfFrames);
 
  unsigned int framesThatWillBeHidden=0;
  for (unsigned int fID=0; fID<mc->numberOfFrames; fID++)
  {
     unsigned int mIDOffset = fID * mc->numberOfValuesPerFrame;

     //Ignore initial xyz..
     for (unsigned int mID=6; mID<mc->numberOfValuesPerFrame; mID++)
     {
        #define USE_SIMPLE_THRESHOLD 0

        #if USE_SIMPLE_THRESHOLD
          float minThreshold = 90.0 - threshold;
          float maxThreshold = 90.0 + threshold;
          float value = mc->motionValues[mIDOffset + mID];
          if  (
               (minThreshold < value) && (value < maxThreshold)
              )
         {
          framesToRemove[fID]=1;
          ++framesThatWillBeHidden;
         }
        #else
          float minThreshold = threshold;
          float maxThreshold = 90.0 - threshold;
          float value = mc->motionValues[mIDOffset + mID];
          float modValue = fmod(fabs(value),90.0);

          if  (
                (value>45.0) && ( ( modValue <= minThreshold) || (maxThreshold <= modValue) )
              )
         {
          //fprintf(stderr,"Value %0.2f offends gimbal rules\n",value);
          framesToRemove[fID]=1;
          ++framesThatWillBeHidden;
          break;
         }
        #endif // USE_SIMPLE_THRESHOLD


        //fprintf(stderr,"fID %u, mID %u => %0.2f  ",fID,mID);

     }

  }

 if (framesThatWillBeHidden==0)
  {
    fprintf(stderr,GREEN "No frames matches our gimbal check rules ( 90 +- %0.2f )\n" NORMAL,threshold);
  } else
  {
    fprintf(stderr,YELLOW "%u/%u (%0.2f%%) frames match gimbal check rules and will be hidden..\n" NORMAL,framesThatWillBeHidden,mc->numberOfFrames,(float) (framesThatWillBeHidden*100)/mc->numberOfFrames);
    fprintf(stderr,GREEN "BVH had %u frames and now " NORMAL,mc->numberOfFrames);
    bvh_removeSelectedFrames(mc,framesToRemove);
    fprintf(stderr,GREEN "it has %u frames \n" NORMAL,mc->numberOfFrames);
  }

 //Free temporary space required to select correct frames..
 free(framesToRemove);

 return 1;
}

















int filterOutPosesThatAreCloseToRules(struct BVH_MotionCapture * mc,int argc,const char **argv)
{
/*
 for (int i=0; i<argc; i++)
 {
   fprintf(stderr,"%02u| %s \n",i,argv[i]);
 }
*/
 if (argc<11)
 {
   fprintf(stderr,"filterOutPosesThatAreCloseToRules: Too few arguments..\n");
   return 0;
 }

 unsigned int width=atoi(argv[6]);
 unsigned int height=atoi(argv[7]);
 float fX=atof(argv[8]);
 float fY=atof(argv[9]);

 unsigned int numberOfRules=atoi(argv[10]);

 if (argc<11+numberOfRules*4)
 {
   fprintf(stderr,"filterOutPosesThatAreCloseToRules: Too few arguments at rules..\n");
   return 0;
 }

  unsigned int * framesToRemove = (unsigned int * ) malloc(sizeof(unsigned int) * mc->numberOfFrames);

  if (framesToRemove==0)
  {
    fprintf(stderr,"Could not allocate space to accomodate poses..\n");
    return 0;
  }

  memset(framesToRemove,0,sizeof(unsigned int) * mc->numberOfFrames);


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
  forcePosition[0]=atof(argv[0]);
  forcePosition[1]=atof(argv[1]);
  forcePosition[2]=atof(argv[2]);
  float forceRotation[3]={0.0,0.0,0.0};
  forceRotation[0]=atof(argv[3]);
  forceRotation[1]=atof(argv[4]);
  forceRotation[2]=atof(argv[5]);

  struct BVH_Transform bvhTransform={0};
  unsigned int fID=0;
  unsigned int framesThatWillBeHidden=0;
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
      const char * jointA = argv[11+rule*4+0];
      BVHJointID jIDA=0;
      const char * jointB = argv[11+rule*4+1];
      BVHJointID jIDB=0;
      float  minimumDistance = atof(argv[11+rule*4+2]);
      float  maximumDistance = atof(argv[11+rule*4+3]);

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
           //fprintf(stderr,"Frame %u ||| %s -> %s =|> min(%0.2f) max(%0.2f)\n",fID,jointA,jointB,minimumDistance,maximumDistance);
           float value = getDistanceBetweenJoints(&bvhTransform,&jIDA,&jIDB);

           if (
                (minimumDistance<value) &&
                (value<maximumDistance)
              )
               {
                 ++rulesThatApplyForFrame;
                 //fprintf(stderr,"Partial hit for Frame %u\n",fID);
               }
         }
     }

     if (rulesThatApplyForFrame==numberOfRules)
     {
       //Frame matches our rules so we must hide it..!
       //fprintf(stderr,"Frame %u Matches rules and will be hidden..\n",fID);
       framesToRemove[fID]=1;
       ++framesThatWillBeHidden;
     }

      bvh_cleanTransform(mc,&bvhTransform);
  }
 }

 if (framesThatWillBeHidden==0)
  {
    fprintf(stderr,GREEN "No frames matches our rules\n" NORMAL);
  } else
  {
    fprintf(stderr,GREEN "%u/%u (%0.2f%%) frames match rules and will be hidden..\n" NORMAL,framesThatWillBeHidden,mc->numberOfFrames,(float) (framesThatWillBeHidden*100)/mc->numberOfFrames);
    fprintf(stderr,GREEN "BVH had %u frames and now " NORMAL,mc->numberOfFrames);
    bvh_removeSelectedFrames(mc,framesToRemove);
    fprintf(stderr,GREEN "it has %u frames \n" NORMAL,mc->numberOfFrames);
  }


 bvh_freeTransform(&bvhTransform);
 free(framesToRemove);

 return 1;
}







int probeForFilterRules(struct BVH_MotionCapture * mc,int argc,const char **argv)
{
 for (unsigned int i=0; i<argc; i++)
 {
   fprintf(stderr,"%02u| %s \n",i,argv[i]);
 }

 if (argc<11)
 {
   fprintf(stderr,"probeForFilterRules: Too few arguments..\n");
   return 0;
 }

 unsigned int width=atoi(argv[6]);
 unsigned int height=atoi(argv[7]);
 float fX=atof(argv[8]);
 float fY=atof(argv[9]);

 unsigned int numberOfRules=atoi(argv[10]);

 if (argc<11+numberOfRules*4)
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
  forcePosition[0]=atof(argv[0]);
  forcePosition[1]=atof(argv[1]);
  forcePosition[2]=atof(argv[2]);
  float forceRotation[3]={0.0,0.0,0.0};
  forceRotation[0]=atof(argv[3]);
  forceRotation[1]=atof(argv[4]);
  forceRotation[2]=atof(argv[5]);



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
      const char * jointA = argv[11+rule*4+0];
      BVHJointID jIDA=0;
      const char * jointB = argv[11+rule*4+1];
      BVHJointID jIDB=0;
      float  minimumDistance = atof(argv[11+rule*4+2]);
      float  maximumDistance = atof(argv[11+rule*4+3]);

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

           if (
                (minimumDistance<value) &&
                (value<maximumDistance)
              )
               {
                 ++rulesThatApplyForFrame;
                 //fprintf(stderr,"Partial hit for Frame %u\n",fID);
               }

         }
     } //Apply each rule

     if (rulesThatApplyForFrame==numberOfRules)
     {
       //Frame matches our rules so we must hide it..!
       fprintf(stderr,GREEN "Frame %u Matches..\n" NORMAL,fID);
     }

  } //Transform correctly calculated..
 } //End of loop for every BVH frame


 return 1;
}

