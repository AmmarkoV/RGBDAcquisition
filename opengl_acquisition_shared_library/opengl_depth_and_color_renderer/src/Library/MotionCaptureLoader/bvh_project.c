#include <stdio.h>

#include "bvh_loader.h"
#include "bvh_project.h"

void bvh_cleanTransform(
                       struct BVH_MotionCapture * mc,
                       struct BVH_Transform     * bvhTransform
                      )
{
 unsigned int jID=0;
 for (jID=0; jID<mc->jointHierarchySize; jID++)
         {
          bvhTransform->joint[jID].pos3D[0]=0.0;
          bvhTransform->joint[jID].pos3D[1]=0.0;
          bvhTransform->joint[jID].pos3D[2]=0.0;

          bvhTransform->joint[jID].pos2DCalculated=0;
          bvhTransform->joint[jID].isBehindCamera=0;

          bvhTransform->joint[jID].pos2D[0]=0.0;
          bvhTransform->joint[jID].pos2D[1]=0.0;
         }
}


int bvh_projectTo2D(
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform     * bvhTransform,
                     struct simpleRenderer    * renderer,
                     unsigned int               occlusions
                   )
{
      unsigned int pointsDumped=0;
      float position2DX;
      float position2DY;
      float position2DW;


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

   //#define DO_TEST 0

   #if DO_TEST
           deadSimpleRendererRender(
                                     renderer,
                                     pos3DFloat,
                                     &position2DX,
                                     &position2DY,
                                     &position2DW
                                   );
   #else
           float pos3DCenterFloat[4];
           pos3DCenterFloat[0]=(float)bvhTransform->centerPosition[0];
           pos3DCenterFloat[1]=(float)bvhTransform->centerPosition[1];
           pos3DCenterFloat[2]=(float)bvhTransform->centerPosition[2];
           pos3DCenterFloat[3]=0.0;

           simpleRendererRender(
                                 renderer ,
                                 pos3DFloat,
                                 pos3DCenterFloat,
                                 0,
                                 0,
                                 &pos3DCenterFloat[0],
                                 &pos3DCenterFloat[1],
                                 &pos3DCenterFloat[2],
                                 &position2DX,
                                 &position2DY,
                                 &position2DW
                                );

            //Should this only happen when position2DW>=0.0
            bvhTransform->joint[jID].pos3D[0] = (double) pos3DCenterFloat[0];
            bvhTransform->joint[jID].pos3D[1] = (double) pos3DCenterFloat[1];
            bvhTransform->joint[jID].pos3D[2] = (double) pos3DCenterFloat[2];
    #endif // DO_TEST

           if (position2DW<0.0)
           {
              bvhTransform->joint[jID].pos2D[0] = 0.0;
              bvhTransform->joint[jID].pos2D[1] = 0.0;
              bvhTransform->joint[jID].isBehindCamera=1;
           } else
           {
              bvhTransform->joint[jID].pos2D[0] = (double) position2DX;
              bvhTransform->joint[jID].pos2D[1] = (double) position2DY;
              bvhTransform->joint[jID].pos2DCalculated=1;
           }

        ++pointsDumped;
      } //Joint Loop


 /*
 if (occlusions)
     {
       //bvhTransform->joint[jID].isOccluded=0;
       unsigned int jID2=0;
       for (jID=0; jID<mc->jointHierarchySize; jID++)
       {
        if (bvhTransform->joint[jID].pos2DCalculated)
        {
         for (jID2=0; jID2<mc->jointHierarchySize; jID2++)
          {
           if (jID2!=jID)
           {
            if (bvhTransform->joint[jID2].pos2DCalculated)
             {
             //
             }
           }
          }
        }
       }
     }
*/
 return (pointsDumped==mc->jointHierarchySize);
}
