#include "bvh_to_trajectoryParserPrimitives.h"

#include <stdio.h>
#include <string.h>
#include "../bvh_transform.h"



void dumpSphereHeader(struct BVH_MotionCapture * mc,FILE *fp)
{
 unsigned int jID;
    for (jID=0; jID<mc->jointHierarchySize; jID++)
    {
      if ( mc->jointHierarchy[jID].isEndSite )  { fprintf(fp,"OBJECT_TYPE(sT%u,cube)\n",jID);   } else
                                                { fprintf(fp,"OBJECT_TYPE(sT%u,sphere)\n",jID); }

      if ( mc->jointHierarchy[jID].isEndSite )  { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 0,255,0,0,0 ,3.0,3.0,3.0)\n",jID,jID);   } else
      if ( mc->jointHierarchy[jID].isRoot )     { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 255,255,0,0,0,4.5,4.5,4.5)\n",jID,jID); } else
                                                { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 255,0,0,0,0 ,2.5,2.5,2.5)\n",jID,jID);   }


      if (bhv_jointHasParent(mc,jID))
      {
        fprintf(fp,"CONNECTOR(s%u,s%u, 255,255,0,100, 3.5)\n",jID,mc->jointHierarchy[jID].parentJoint);
      }
    }
    fprintf(fp,"\n");
}


void dumpSphereBody(
                    struct BVH_MotionCapture * mc,
                    struct BVH_Transform * bvhTransform,
                    FILE *fp ,
                    unsigned int fID
                   )
{
     unsigned int jID=0;
     bvh_loadTransformForFrame(
                                mc,
                                fID ,
                                bvhTransform
                              );

     fprintf(fp,"POS(camera,%u,60.0,60.0,252.0,0.0,0.0,0.0,0.0)\n",fID);
     fprintf(fp,"POS(floor,%u,0.0,0.0,0.0,0.0,0.0,0.0,0.0)\n",fID);
     for (jID=0; jID<mc->jointHierarchySize; jID++)
     {
      fprintf(
              fp,"POS(s%u,%u,%0.4f,%0.4f,%0.4f,0,0,0,0)\n",jID,fID,
              bvhTransform->joint[jID].pos3D[0],
              bvhTransform->joint[jID].pos3D[1],
              bvhTransform->joint[jID].pos3D[2]
             );
     }
     fprintf(fp,"\n");
}





int dumpBVHToTrajectoryParserPrimitives(const char * filename , struct BVH_MotionCapture * mc)
{
  unsigned int fID=0;
  FILE * fp = fopen(filename,"w");

  struct BVH_Transform bvhTransform={0};

  if (fp!=0)
  {
    fprintf(fp,"#Auto generated using BVHTester to render file : %s to scene : %s\n",mc->fileName,filename);
    fprintf(fp,"#https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer\n");
    fprintf(fp,"AUTOREFRESH(1500)\n");
    fprintf(fp,"BACKGROUND(63,114,182)\n");


    //fprintf(fp,"SCALE_WORLD(-1,1,1)\n");
    fprintf(fp,"NEAR_CLIP(0.1)\n");
    fprintf(fp,"FAR_CLIP(1000)\n");
    fprintf(fp,"SILENT(1)\n");
    fprintf(fp,"RATE(120)\n");
    fprintf(fp,"INTERPOLATE_TIME(0)\n");
    fprintf(fp,"MOVE_VIEW(1)\n\n");


    fprintf(fp,"OBJECT_TYPE(floorType,grid)\n");
    fprintf(fp,"OBJECT(floor,floorType,0,255,0,0 ,0, 10.0,10.0,10.0)\n");
    //Instantiate objects that will draw our skeleton
    //------------------------------------------------
      dumpSphereHeader(mc,fp);
    //------------------------------------------------

    for (fID=0; fID<mc->numberOfFrames; fID++)
    {
      dumpSphereBody(
                     mc,
                     &bvhTransform,
                     fp ,
                     fID
                    );
    }

    fclose(fp);
    return 1;
  }
 return 0;
}
