#include "bvh_to_trajectoryParserPrimitives.h"

#include <stdio.h>
#include <string.h>
#include "../calculate/bvh_transform.h"



void dumpSphereHeader(struct BVH_MotionCapture * mc,FILE *fp)
{
 unsigned int jID;

    for (jID=0; jID<mc->jointHierarchySize; jID++)
    {
      if ( mc->jointHierarchy[jID].isEndSite )  { fprintf(fp,"OBJECT_TYPE(sT%u,cube)\n",jID);   } else
                                                { fprintf(fp,"OBJECT_TYPE(sT%u,sphere)\n",jID); }

      if ( mc->jointHierarchy[jID].isEndSite )  { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 0,248,138,35,0 ,3.0,3.0,3.0)\n",jID,jID);   } else
      if ( mc->jointHierarchy[jID].isRoot )     { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 255,255,0,0,0,4.5,4.5,4.5)\n",jID,jID); } else
                                                { fprintf(fp,"RIGID_OBJECT(s%u,sT%u, 205,35,240,0,0 ,2.5,2.5,2.5)\n",jID,jID);   }


      unsigned int connectorColorR=255;
      unsigned int connectorColorG=255;
      unsigned int connectorColorB=0;
      if ( mc->jointHierarchy[jID].partOfHierarchy.isAPartOfRightFoot ) { connectorColorR=0; connectorColorG=255; connectorColorB=0; } else
      if ( mc->jointHierarchy[jID].partOfHierarchy.isAPartOfRightArm )  { connectorColorR=0; connectorColorG=255; connectorColorB=0; } else
      if ( mc->jointHierarchy[jID].partOfHierarchy.isAPartOfLeftFoot )  { connectorColorR=255; connectorColorG=0; connectorColorB=0; } else
      if ( mc->jointHierarchy[jID].partOfHierarchy.isAPartOfLeftArm )   { connectorColorR=255; connectorColorG=0; connectorColorB=0; } else
      if ( mc->jointHierarchy[jID].partOfHierarchy.isAPartOfHead)       { connectorColorR=0; connectorColorG=0; connectorColorB=255; } else
      if ( mc->jointHierarchy[jID].partOfHierarchy.isAPartOfTorso )     { connectorColorR=0; connectorColorG=0; connectorColorB=255; }


      if (bhv_jointHasParent(mc,jID))
      {
        fprintf(fp,"CONNECTOR(s%u,s%u,%u,%u,%u,100, 3.5)\n",jID,mc->jointHierarchy[jID].parentJoint,connectorColorR,connectorColorG,connectorColorB);
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
   if (
        bvh_loadTransformForFrame(
                                  mc,
                                  fID ,
                                  bvhTransform
                                  ,1/*Populate extra structures*/
                                 )
       )
    {
     fprintf(fp,"POS(camera,%u,60.0,0.0,252.0,0.0,0.0,0.0,0.0)\n",fID);
     //fprintf(fp,"POS(floor,%u,0.0,0.0,0.0,0.0,0.0,0.0,0.0)\n",fID);

     unsigned int jID=0;
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
     
     bvh_freeTransform(bvhTransform);
    }
}





int dumpBVHToTrajectoryParserPrimitives(const char * filename , struct BVH_MotionCapture * mc)
{
  FILE * fp = fopen(filename,"w");

  struct BVH_Transform bvhTransform={0};

  if (fp!=0)
  {
    fprintf(fp,"#Auto generated using BVHTester to render file : %s to scene : %s\n",mc->fileName,filename);
    fprintf(fp,"#https://github.com/AmmarkoV/RGBDAcquisition/tree/master/opengl_acquisition_shared_library/opengl_depth_and_color_renderer\n");
    fprintf(fp,"AUTOREFRESH(1500)\n");
    fprintf(fp,"#BACKGROUND(63,114,182)\n");
    fprintf(fp,"BACKGROUND(0,0,0)\n");


    //fprintf(fp,"SCALE_WORLD(-1,1,1)\n");
    fprintf(fp,"NEAR_CLIP(0.1)\n");
    fprintf(fp,"FAR_CLIP(1000)\n");
    fprintf(fp,"SILENT(1)\n");
    fprintf(fp,"RATE(120)\n");
    fprintf(fp,"INTERPOLATE_TIME(1)\n");
    fprintf(fp,"MOVE_VIEW(1)\n\n");


    //fprintf(fp,"OBJECT_TYPE(floorType,grid)\n");
    //fprintf(fp,"OBJECT(floor,floorType,0,255,0,0 ,0, 10.0,10.0,10.0)\n");
    //Instantiate objects that will draw our skeleton
    //------------------------------------------------
      dumpSphereHeader(mc,fp);
    //------------------------------------------------

    unsigned int fID=0;
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
