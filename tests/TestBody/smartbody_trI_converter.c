#include "smartbody_trI_converter.h"


int convertCOCO_To_Smartbody_TRI(struct skeletonCOCO * coco,struct TRI_Model * triModel ,
                                 float *x , float *y , float *z ,
                                 float *qX,float *qY,float *qZ,float *qW )
{
  if (coco->joint[COCO_LHip].x < coco->joint[COCO_RHip].x) { *x=coco->joint[COCO_LHip].x; } else { *x=coco->joint[COCO_RHip].x; }
  if (coco->joint[COCO_LHip].y < coco->joint[COCO_RHip].y) { *y=coco->joint[COCO_LHip].y; } else { *y=coco->joint[COCO_RHip].y; }
  if (coco->joint[COCO_LHip].z < coco->joint[COCO_RHip].z) { *z=coco->joint[COCO_LHip].z; } else { *z=coco->joint[COCO_RHip].z; }

  *x+=(coco->joint[COCO_LHip].x-coco->joint[COCO_RHip].x)/2;
  *y+=(coco->joint[COCO_LHip].y-coco->joint[COCO_RHip].y)/2;
  *z+=(coco->joint[COCO_LHip].z-coco->joint[COCO_RHip].z)/2;
  *qX=0.707107;
  *qY=0.707107;
  *qZ=0.000000;
  *qW=0.0;
 return 0;
}
