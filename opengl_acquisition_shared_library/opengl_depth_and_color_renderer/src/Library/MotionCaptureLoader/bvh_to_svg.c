#include <stdio.h>
#include "bvh_to_svg.h"
#include "../../../../../tools/AmMatrix/matrix4x4Tools.h"
#include "../../../../../tools/AmMatrix/matrixOpenGL.h"



int dumpBVHToSVGFile(
                     const char * filename,
                     struct BVH_MotionCapture * mc,
                     struct BVH_Transform * bvhTransform,
                     unsigned int fID,
                     ///The rest are matrices to do projections..
                     float * viewMatrix,
                     float * projectionMatrix,
                     int * viewport
                    )
{
   unsigned int width = viewport[2];
   unsigned int height = viewport[3];

   float windowCoordinates[3]={0};
   float  modelViewMatrix[16];

   FILE * fp = fopen(filename,"w");
   if (fp!=0)
   {
      fprintf(fp,"<svg width=\"%u\" height=\"%u\">\n",width,height);
      fprintf(fp,"<rect width=\"%u\" height=\"%u\" style=\"fill:rgb(255,255,255);stroke-width:3;stroke:rgb(255,255,255)\" />\n",width,height);
      fprintf(fp,"<text x=\"10\" y=\"40\">Frame %u</text>\n",fID);


      //First load the 3D positions of each joint..
      bvh_loadTransformForFrame(
                                mc,
                                fID ,
                                bvhTransform
                               );
      //Then project 3D positions on 2D frame and save results..
      unsigned int jID=0;
      for (jID=0; jID<mc->jointHierarchySize; jID++)
      {

        if (bvhTransform->joint[jID].pos3D[3]!=1.0)
        {
          fprintf(stderr,"bvh_loadTransformForFrame location for joint %u not normalized..\n",jID);
        }

           float position3DX = bvhTransform->joint[jID].pos3D[0];
           float position3DY = bvhTransform->joint[jID].pos3D[1];
           float position3DZ = bvhTransform->joint[jID].pos3D[2];

            double modelTransformationD[16];
            float  modelTransformationF[16];
            create4x4ModelTransformation(
                                           modelTransformationD,
                                           //Rotation Component
                                           (double) 0.0,//heading,
                                           (double) 0.0,//pitch,
                                           (double) 0.0,//roll,
                                           ROTATION_ORDER_RPY,
                                           //Translation Component
                                           (double) position3DX,
                                           (double) position3DY,
                                           (double) position3DZ,
                                           //Scale Component
                                           (double) 1.0,
                                           (double) 1.0,
                                           (double) 1.0
                                         );
         copy4x4DMatrixToF(modelTransformationF,modelTransformationD);
         multiplyTwo4x4FMatrices(modelViewMatrix,viewMatrix,modelTransformationF);


         _glhProjectf(
                      position3DX,
                      position3DX,
                      position3DZ,
                      modelViewMatrix,
                      projectionMatrix,
                      viewport,
                      windowCoordinates
                     );

        fprintf(fp,"<circle cx=\"%0.2f\" cy=\"%0.2f\" r=\"8.00\" stroke=\"rgb(135,135,0)\" stroke-width=\"3\" fill=\"rgb(255,255,0)\" />\n",
                  windowCoordinates[0],
                  windowCoordinates[1]);
      }


      fprintf(fp,"</svg>\n");
     return 1;
   }
 return 0;
}







int dumpBVHToSVG(
                 const char * directory ,
                 struct BVH_MotionCapture * mc,
                 unsigned int width,
                 unsigned int height
                 )
{
  struct BVH_Transform bvhTransform;
  char filename[512];

  unsigned int framesDumped=0;
  unsigned int fID=0;


 float projectionMatrix[16];
 float viewMatrix[16];
 int   viewport[4]={0};
 viewport[2]=width;
 viewport[3]=height;



  float fx = 575.816;
  float fy = 575.816;
  float skew = 0.0;
  float cx = (float) width/2;
  float cy = (float) height/2;
  float near = 1.0;
  float far = 1000.0;

  buildOpenGLProjectionForIntrinsics(
                                      projectionMatrix ,
                                      viewport ,
                                      fx,
                                      fy,
                                      skew,
                                      cx,
                                      cy,
                                      width,
                                      height,
                                      near,
                                      far
                                     );

/*
  glhPerspectivef2(
                   projectionMatrix,
                   65,//fovyInDegrees,
                   (float) width/height,//aspectRatioV,
                   1.0,//znear,
                   1000//zfar
                  );*/

   double viewMatrixD[16];
   create4x4ScalingMatrix(viewMatrixD,-1.0,1.0,1.0);
   copy4x4DMatrixToF(viewMatrix,viewMatrixD);

   double viewportMatrixD[16];
   glGetViewportMatrix(viewportMatrixD, viewport[0],viewport[1],viewport[2],viewport[3],(double) near,(double) far);




  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   snprintf(filename,512,"%s/%06u.svg",directory,fID);

   framesDumped +=  dumpBVHToSVGFile(
                                     filename,
                                     mc,
                                     &bvhTransform,
                                     fID,
                                     viewMatrix,
                                     projectionMatrix,
                                     viewport
                                    );
  }

 return (framesDumped==mc->numberOfFrames);
}
