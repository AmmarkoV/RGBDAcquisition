#include "smartbody_trI_converter.h"
#include "../../opengl_acquisition_shared_library/opengl_depth_and_color_renderer/src/ModelLoader/model_loader_transform_joints.h"
#include "../../tools/AmMatrix/matrixProject.h"


float absF(float abs)
{
 if(abs<0) { return -abs; }
 return abs;
}



float getCOCOAndSmartBodyDistance(struct skeletonCOCO * coco,struct TRI_Model * triModel)
{
  float score=10000000;

  unsigned int outputNumberOfJoints;
  float * triJoints = convertTRIBonesToJointPositions(triModel,&outputNumberOfJoints);
  if (triJoints!=0)
  {
   score=0;
    unsigned int i=0;
    for (i=0; i<COCO_PARTS; i++)
    {
      unsigned triJointAddr;
      if ( findTRIBoneWithName(triModel ,smartBodyNames[i], &triJointAddr) )
      {
       float diffX = coco->joint[COCO_LHip].x - triModel->joint[triJointAddr*3+0];
       float diffY = coco->joint[COCO_LHip].y - triModel->joint[triJointAddr*3+1];
       float diffZ = coco->joint[COCO_LHip].z - triModel->joint[triJointAddr*3+2];
       score+=sqrt((diffX*diffX)+(diffY*diffY)+(diffZ*diffZ));
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
  if (coco->joint[COCO_LHip].x < coco->joint[COCO_RHip].x) { *x=coco->joint[COCO_LHip].x; } else { *x=coco->joint[COCO_RHip].x; }
  if (coco->joint[COCO_LHip].y < coco->joint[COCO_RHip].y) { *y=coco->joint[COCO_LHip].y; } else { *y=coco->joint[COCO_RHip].y; }
  if (coco->joint[COCO_LHip].z < coco->joint[COCO_RHip].z) { *z=coco->joint[COCO_LHip].z; } else { *z=coco->joint[COCO_RHip].z; }

  *x+=absF((coco->joint[COCO_LHip].x-coco->joint[COCO_RHip].x)/2);
  *y+=absF((coco->joint[COCO_LHip].y-coco->joint[COCO_RHip].y)/2);
  *z+=absF((coco->joint[COCO_LHip].z-coco->joint[COCO_RHip].z)/2);

  unsigned int outputNumberOfJoints;
  float * triJoints = convertTRIBonesToJointPositions(triModel,&outputNumberOfJoints);
  if (triJoints!=0)
  {
    unsigned int  * verticesToKeep = getClosestVertexToJointPosition(triModel,triJoints,outputNumberOfJoints);
    if (verticesToKeep!=0)
    {
      free(verticesToKeep);
    }
    free(triJoints);
  }




  *qX=0.707107;
  *qY=0.707107;
  *qZ=0.000000;
  *qW=0.0;
 return 0;
}
