#include "simpleRenderer.h"
#include "matrix4x4Tools.h"
#include "matrixOpenGL.h"
#include <stdio.h>


int simpleRendererRender(
                         struct simpleRenderer * sr ,
                         float * position3D,
                         float * output2DX,
                         float * output2DY
                        )
{
 double modelTransformationD[16];
 float  modelTransformationF[16];
 double position3DX = position3D[0];
 double position3DY = position3D[1];
 double position3DZ = position3D[2];
 ///--------------------------------------------------------------------
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
 ///--------------------------------------------------------------------
 copy4x4DMatrixToF(modelTransformationF,modelTransformationD);
 multiplyTwo4x4FMatrices(sr->modelViewMatrix,sr->viewMatrix,modelTransformationF);


 ///--------------------------------------------------------------------
  float windowCoordinates[3]={0};
  _glhProjectf(
                position3D,
                sr->modelViewMatrix,
                sr->projectionMatrix,
                sr->viewport,
                windowCoordinates
              );
  //fprintf(stderr,"(%0.2f,%0.2f,%0.2f->%0.2f,%0.2f)",position3DX,position3DX,position3DZ,windowCoordinates[0],windowCoordinates[2]);

 ///--------------------------------------------------------------------
  *output2DX = windowCoordinates[0];//windowCoordinates[2];
  *output2DY = windowCoordinates[1];//windowCoordinates[2];
  return 1;
}






int simpleRendererInitialize(struct simpleRenderer * sr)
{
  sr->viewport[0]=0;
  sr->viewport[1]=0;
  sr->viewport[2]=sr->width;
  sr->viewport[3]=sr->height;

  buildOpenGLProjectionForIntrinsics(
                                      sr->projectionMatrix ,
                                      sr->viewport ,
                                      sr->fx,
                                      sr->fy,
                                      sr->skew,
                                      sr->cx,
                                      sr->cy,
                                      sr->width,
                                      sr->height,
                                      sr->near,
                                      sr->far
                                     );


/*
  glhPerspectivef2(
                   sr->projectionMatrix,
                   65,//fovyInDegrees,
                   (float) sr->width/sr->height,//aspectRatioV,
                   sr->near,
                   sr->far
                  );*/


   double viewMatrixD[16];
   create4x4ScalingMatrix(viewMatrixD,-1.0,1.0,1.0);
   copy4x4DMatrixToF(sr->viewMatrix,viewMatrixD);
   create4x4IdentityMatrixF(sr->viewMatrix);

   //Model
   create4x4IdentityMatrixF(sr->modelMatrix);

 return 1;
}
