#include <stdio.h>

#include "bvh_project.h"

int bvh_projectTo2D(
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform     * bvhTransform,
                     struct simpleRenderer    * renderer,
                     float * objectRotation
                   )
{
      unsigned int pointsDumped=0;
      float position2DX;
      float position2DY;

      //Then project 3D positions on 2D frame and save results..
      unsigned int jID=0;
      for (jID=0; jID<mc->jointHierarchySize; jID++)
      {
        if (bvhTransform->joint[jID].pos3D[3]!=1.0)
           { fprintf(stderr,"bvh_loadTransformForFrame location for joint %u not normalized..\n",jID); }

           float pos3DFloat[4];
           pos3DFloat[0]=(float)bvhTransform->joint[jID].pos3D[0];
           pos3DFloat[1]=(float)bvhTransform->joint[jID].pos3D[1];
           pos3DFloat[2]=(float)bvhTransform->joint[jID].pos3D[2];
           pos3DFloat[3]=0.0;

           float pos3DCenterFloat[4];
           pos3DCenterFloat[0]=(float)bvhTransform->centerPosition[0];
           pos3DCenterFloat[1]=(float)bvhTransform->centerPosition[1];
           pos3DCenterFloat[2]=(float)bvhTransform->centerPosition[2];
           pos3DCenterFloat[3]=0.0;

           simpleRendererRender(
                                 renderer ,
                                 pos3DFloat,
                                 pos3DCenterFloat,
                                 objectRotation,
                                 &pos3DCenterFloat[0],
                                 &pos3DCenterFloat[1],
                                 &pos3DCenterFloat[2],
                                 &position2DX,
                                 &position2DY
                                );

           bvhTransform->joint[jID].pos3D[0] = (double) pos3DCenterFloat[0];
           bvhTransform->joint[jID].pos3D[1] = (double) pos3DCenterFloat[1];
           bvhTransform->joint[jID].pos3D[2] = (double) pos3DCenterFloat[2];

           bvhTransform->joint[jID].pos2D[0] = (double) position2DX;
           bvhTransform->joint[jID].pos2D[1] = (double) position2DY;
           bvhTransform->joint[jID].pos2DCalculated=1;
           ++pointsDumped;
      } //Joint Loop


 return (pointsDumped==mc->jointHierarchySize);
}
