#include "simpleRenderer.h"
#include "matrix4x4Tools.h"
#include "matrixOpenGL.h"



int simpleRendererRender(
                         struct simpleRenderer * sr ,
                         float position3DX,
                         float position3DY,
                         float position3DZ,
                         float * output2DX,
                         float * output2DY
                        )
{
 double modelTransformationD[16];
 float  modelTransformationF[16];
 float  modelViewMatrix[16];
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
 multiplyTwo4x4FMatrices(modelViewMatrix,sr->viewMatrix,modelTransformationF);


 ///--------------------------------------------------------------------
  float windowCoordinates[3]={0};
  _glhProjectf(
                position3DX,
                position3DX,
                position3DZ,
                modelViewMatrix,
                sr->projectionMatrix,
                sr->viewport,
                windowCoordinates
              );
 ///--------------------------------------------------------------------
  *output2DX = windowCoordinates[0];
  *output2DY = windowCoordinates[1];
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
                   projectionMatrix,
                   65,//fovyInDegrees,
                   (float) width/height,//aspectRatioV,
                   1.0,//znear,
                   1000//zfar
                  );*/


   double viewMatrixD[16];
   create4x4ScalingMatrix(viewMatrixD,-1.0,1.0,1.0);
   copy4x4DMatrixToF(sr->viewMatrix,viewMatrixD);

   //create4x4IdentityMatrixF(sr->viewMatrix);
 return 0;
}
