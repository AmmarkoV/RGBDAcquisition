#include "smartbody_tri_converter.h"
#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_loader_transform_joints.h"
#include "../../tools/AmMatrix/matrixProject.h"


#include <sys/time.h>
#include <time.h>

unsigned long tickBase = 0;

unsigned long getTickCountMicroseconds()
{
   struct timespec ts;
   if ( clock_gettime(CLOCK_MONOTONIC,&ts) != 0) { return 0; }

   if (tickBase==0)
   {
     tickBase = ts.tv_sec*1000000 + ts.tv_nsec/1000;
     return 0;
   }

   return ( ts.tv_sec*1000000 + ts.tv_nsec/1000 ) - tickBase;
}

float absF(float abs)
{
 if(abs<0) { return -abs; }
 return abs;
}

float getCOCOAndSmartBodyDistance(struct skeletonCOCO * coco,struct TRI_Model * triModel)
{
  float score=10000000,addedToScore=0;

  unsigned int outputNumberOfJoints;
  float * triJoints = convertTRIBonesToJointPositions(triModel,&outputNumberOfJoints);
  if (triJoints!=0)
  {
   score=0;
    unsigned int i=0;
    unsigned triJointAddr=0;
    for (i=0; i<COCO_PARTS; i++)
    {
     if (cocoMapToSmartBody[i]<HUMAN_SKELETON_PARTS)
     {
      if ( findTRIBoneWithName(triModel ,smartBodyNames[cocoMapToSmartBody[i]], &triJointAddr) )
      {
       float diffX = coco->joint[i].x - triJoints[triJointAddr*3+0];
       float diffY = coco->joint[i].y - triJoints[triJointAddr*3+1];
       float diffZ = coco->joint[i].z - triJoints[triJointAddr*3+2];
       addedToScore=sqrt((diffX*diffX)+(diffY*diffY)+(diffZ*diffZ));
       fprintf(stderr,"Joint %s has diffs = %0.2f " ,smartBodyNames[cocoMapToSmartBody[i]], addedToScore);
       fprintf(stderr,"( %0.2f %0.2f %0.2f ) -> " ,coco->joint[i].x,coco->joint[i].y,coco->joint[i].z);
       fprintf(stderr,"( %0.2f %0.2f %0.2f )  \n" ,triJoints[triJointAddr*3+0],triJoints[triJointAddr*3+1],triJoints[triJointAddr*3+2]);
       score+=addedToScore;
      }
     }
    }

   free(triJoints);
  }

 return score;
}

int convertCOCO_To_Smartbody_TRI(struct skeletonCOCO * coco,struct TRI_Model * triModel ,
                                 float *x , float *y , float *z ,
                                 float *qX,float *qY,float *qZ,float *qW )
{
 fprintf(stderr,"convertCOCO_To_Smartbody_TRI \n");

 unsigned int jointDataSizeOutput=0;
 float * jointSolution = mallocModelTransformJoints(
                                                     triModel,
                                                     &jointDataSizeOutput
                                                   );


  if (coco->joint[COCO_LHip].x < coco->joint[COCO_RHip].x) { *x=coco->joint[COCO_LHip].x; } else { *x=coco->joint[COCO_RHip].x; }
  if (coco->joint[COCO_LHip].y < coco->joint[COCO_RHip].y) { *y=coco->joint[COCO_LHip].y; } else { *y=coco->joint[COCO_RHip].y; }
  if (coco->joint[COCO_LHip].z < coco->joint[COCO_RHip].z) { *z=coco->joint[COCO_LHip].z; } else { *z=coco->joint[COCO_RHip].z; }

  *x+=absF((coco->joint[COCO_LHip].x-coco->joint[COCO_RHip].x)/2);
  *y+=absF((coco->joint[COCO_LHip].y-coco->joint[COCO_RHip].y)/2);
  *z+=absF((coco->joint[COCO_LHip].z-coco->joint[COCO_RHip].z)/2);


  float currentSolution=0;
  float bestSolution=1000000;
  unsigned int jointToChange=0;
  findTRIBoneWithName(triModel ,smartBodyNames[HUMAN_SKELETON_RIGHT_SHOULDER] , &jointToChange);


  unsigned long startTime,endTime;
  float  bestSelection=666;
  unsigned int i=0;

  float testI=0;
  for (i=0; i<10; i++)
  {
   testI=(float) -50+(i*10);

   startTime = getTickCountMicroseconds();
   fprintf(stderr,"Trying %s joint value %0.2f ",smartBodyNames[HUMAN_SKELETON_RIGHT_SHOULDER],testI);

     transformTRIJoint(
                        triModel,
                        jointSolution,
                        jointDataSizeOutput,

                        jointToChange ,
                        testI,
                        0 ,
                        0
                      );

      struct TRI_Model tmVT = {0};
      doModelTransform(
                         &tmVT , //We won't make an output model..!
                         triModel ,

                         jointSolution,
                         jointDataSizeOutput ,

                         1, //Auto detect default matrices ( speedup )
                         1, //We want direct setting of matrices
                         1,  //We actually don't want to perform the vertex transform we only want the matrices
                         0
                        );
    currentSolution=getCOCOAndSmartBodyDistance(coco,&tmVT);
    deallocInternalsOfModelTri(&tmVT);

   fprintf(stderr,"score=%0.2f ",currentSolution);

    if (currentSolution<bestSolution)
    {
     fprintf(stderr,"\n\nBetter solution found (%0.2f) ..\n",testI);
     fprintf(stderr,"(config %0.2f %0.2f %0.2f ) \n\n",
             triModel->bones[jointToChange].info->rotX,
             triModel->bones[jointToChange].info->rotY,
             triModel->bones[jointToChange].info->rotZ);

     bestSolution=currentSolution;
     bestSelection=testI;
    }

   endTime = getTickCountMicroseconds();
   fprintf(stderr,"%lu microseconds.. \n",endTime-startTime);
  }


  *qX=0.707107;
  *qY=0.707107;
  *qZ=0.000000;
  *qW=0.0;
 return 0;
}
