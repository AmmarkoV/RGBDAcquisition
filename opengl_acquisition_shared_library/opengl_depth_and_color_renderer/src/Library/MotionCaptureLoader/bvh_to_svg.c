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
                     float cameraX,float cameraY,float cameraZ,
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
      fprintf(fp,"<text x=\"10\" y=\"60\">Camera Position(%0.2f,%0.2f,%0.2f)</text>\n",cameraX,cameraY,cameraZ);


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

           float positionX = bvhTransform->joint[jID].pos3D[0];
           float positionY = bvhTransform->joint[jID].pos3D[1];
           float positionZ = bvhTransform->joint[jID].pos3D[2];

            double modelTransformation[16];
            float  modelTransformationF[16];
            create4x4ModelTransformation(
                                           modelTransformation,
                                           //Rotation Component
                                           (double) 0.0,//heading,
                                           (double) 0.0,//pitch,
                                           (double) 0.0,//roll,
                                           ROTATION_ORDER_RPY,
                                           //Translation Component
                                           (double) positionX,
                                           (double) positionY,
                                           (double) positionZ ,
                                           //Scale Component
                                           (double) 1.0,
                                           (double) 1.0,
                                           (double) 1.0
                                         );
         copy4x4DMatrixToF(modelTransformationF,modelTransformation);
         multiplyTwo4x4FMatrices(modelViewMatrix,modelTransformationF,viewMatrix);

         _glhProjectf(
                      bvhTransform->joint[jID].pos3D[0],
                      bvhTransform->joint[jID].pos3D[1],
                      bvhTransform->joint[jID].pos3D[2],
                      modelViewMatrix,
                      projectionMatrix,
                      viewport,
                      windowCoordinates
                     );

        fprintf(fp,"<circle cx=\"%0.2f\" cy=\"%0.2f\" r=\"8.00\" stroke=\"rgb(135,135,0)\" stroke-width=\"3\" fill=\"rgb(255,255,0)\" />\n",windowCoordinates[0],windowCoordinates[1]);
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
 float modelViewMatrix[16];
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

   float cameraX = 60;
   float cameraY = -60;
   float cameraZ = 4300;

   create4x4IdentityMatrixF(modelViewMatrix);
   create4x4FTranslationMatrix(
                               modelViewMatrix,
                               cameraX,
                               cameraY,
                               cameraZ
                               );
   fprintf(stderr,"dumpBVHToSVG: Camera location (%0.2f,%0.2f,%0.2f)\n",cameraX,cameraY,cameraZ);




     create4x4ScalingMatrix(viewMatrixD,-1.0,1.0,1.0);

     glGetViewportMatrix(viewportMatrixD, viewport[0],viewport[1],viewport[2],viewport[3],near,far);




  for (fID=0; fID<mc->numberOfFrames; fID++)
  {
   snprintf(filename,512,"%s/%06u.svg",directory,fID);

   framesDumped +=  dumpBVHToSVGFile(
                                     filename,
                                     mc,
                                     &bvhTransform,
                                     fID,
                                     cameraX,cameraY,cameraZ,
                                     modelViewMatrix,
                                     projectionMatrix,
                                     viewport
                                    );
  }

 return (framesDumped==mc->numberOfFrames);
}
