#include "simpleRenderer.h"
#include "matrix4x4Tools.h"
#include "matrixOpenGL.h"
#include <stdio.h>




int simpleRendererRender(
                         struct simpleRenderer * sr ,
                         float * position3D,
                         float * center3D,
                         float * objectRotation,
                         unsigned int rotationOrder,
                         ///---------------
                         float * output3DX,
                         float * output3DY,
                         float * output3DZ,
                         ///---------------
                         float * output2DX,
                         float * output2DY,
                         float * output2DW
                        )
{
 double modelTransformationD[16];
 float  modelTransformationF[16];
 ///--------------------------------------------------------------------
 ///                       CAMERA MATRICES ETC
 ///--------------------------------------------------------------------
 create4x4ModelTransformation(
                                                        modelTransformationD,
                                                        //Rotation Component
                                                        (double) sr->cameraOffsetRotation[0],//heading,
                                                        (double) sr->cameraOffsetRotation[1],//pitch,
                                                        (double) sr->cameraOffsetRotation[2],//roll,
                                                        ROTATION_ORDER_RPY,
                                                        //Translation Component
                                                        (double) sr->cameraOffsetPosition[0],
                                                        (double) sr->cameraOffsetPosition[1],
                                                        (double) sr->cameraOffsetPosition[2],
                                                        //Scale Component
                                                        (double) 1.0,
                                                        (double) 1.0,
                                                        (double) 1.0
                              );
 ///--------------------------------------------------------------------
 copy4x4DMatrixToF(modelTransformationF,modelTransformationD);
 multiplyTwo4x4FMatrices(sr->modelViewMatrix,sr->viewMatrix,modelTransformationF);


 ///--------------------------------------------------------------------





 ///--------------------------------------------------------------------
 ///                       OBJECT MATRICES ETC
 ///--------------------------------------------------------------------
  double objectMatrixRotation[16];

  if (objectRotation==0)
    {
      create4x4IdentityMatrix(objectMatrixRotation);
    } else
  if ( (objectRotation[0]==0) && (objectRotation[1]==0) && (objectRotation[2]==0) )
    {
      create4x4IdentityMatrix(objectMatrixRotation);
    } else
  if (rotationOrder==ROTATION_ORDER_RPY)
    {
     //This is the old way to do this rotation
     doRPYTransformation(
                         objectMatrixRotation,
                        (double) objectRotation[0],
                        (double) objectRotation[1],
                        (double) objectRotation[2]
                       );
    } else
    {
     //fprintf(stderr,"Using new model transform code\n");
     create4x4MatrixFromEulerAnglesWithRotationOrder(
                                                     objectMatrixRotation ,
                                                     (double) objectRotation[0],
                                                     (double) objectRotation[1],
                                                     (double) objectRotation[2],
                                                     rotationOrder
                                                    );
    }



   double point3D[4];
   double resultPoint3D[4];


 point3D[0]=(double) (position3D[0]-center3D[0]);
 point3D[1]=(double) (position3D[1]-center3D[1]);
 point3D[2]=(double) (position3D[2]-center3D[2]);
 point3D[3]=(double) (1.0);


 transform3DPointVectorUsing4x4Matrix(
                                       resultPoint3D,
                                       objectMatrixRotation,
                                       point3D
                                     );


 float final3DPosition[4];

 if (sr->removeObjectPosition)
  {
   final3DPosition[0]=(float) resultPoint3D[0]+sr->cameraOffsetPosition[0];
   final3DPosition[1]=(float) resultPoint3D[1]+sr->cameraOffsetPosition[1];
   final3DPosition[2]=(float) resultPoint3D[2]+sr->cameraOffsetPosition[2];
  } else
  {
   final3DPosition[0]=(float) resultPoint3D[0]+center3D[0]+sr->cameraOffsetPosition[0];
   final3DPosition[1]=(float) resultPoint3D[1]+center3D[1]+sr->cameraOffsetPosition[1];
   final3DPosition[2]=(float) resultPoint3D[2]+center3D[2]+sr->cameraOffsetPosition[2];
  }
 final3DPosition[3]=(float) 0.0;//resultPoint3D[3];
 ///--------------------------------------------------------------------




 ///--------------------------------------------------------------------
 ///                         FINAL PROJECTION
 ///--------------------------------------------------------------------
  float windowCoordinates[3]={0};

  if (
       !_glhProjectf(
                     final3DPosition,
                     sr->modelViewMatrix,
                     sr->projectionMatrix,
                     sr->viewport,
                     windowCoordinates
                    )
     )
     { fprintf(stderr,"Could not project 3D Point (%0.2f,%0.2f,%0.2f)\n",final3DPosition[0],final3DPosition[1],final3DPosition[2]); }
 ///--------------------------------------------------------------------
  *output2DX = windowCoordinates[0];//windowCoordinates[2];
  *output2DY = windowCoordinates[1];//windowCoordinates[2];
  *output2DW = windowCoordinates[2];
  return 1;
}




int deadSimpleRendererRender(
                             struct simpleRenderer * sr ,
                             float * position3D,
                             ///---------------
                             float * output2DX,
                             float * output2DY,
                             float * output2DW
                            )
{
 ///--------------------------------------------------------------------
 ///                         FINAL PROJECTION
 ///--------------------------------------------------------------------
  float windowCoordinates[3]={0};

  create4x4IdentityMatrixF(sr->modelViewMatrix);

  if (
       !_glhProjectf(
                     position3D,
                     sr->modelViewMatrix,
                     sr->projectionMatrix,
                     sr->viewport,
                     windowCoordinates
                    )
     )
     { fprintf(stderr,"Could not project 3D Point (%0.2f,%0.2f,%0.2f)\n",position3D[0],position3D[1],position3D[2]); }
 ///--------------------------------------------------------------------
  *output2DX = windowCoordinates[0];//windowCoordinates[2];
  *output2DY = windowCoordinates[1];//windowCoordinates[2];
  *output2DW = windowCoordinates[2];
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

   double viewMatrixD[16];
   create4x4IdentityMatrix(viewMatrixD);

   create4x4ScalingMatrix(viewMatrixD,0.01,0.01,-0.01);
   copy4x4DMatrixToF(sr->viewMatrix,viewMatrixD);

   //Initialization of matrices not yet used
   create4x4IdentityMatrixF(sr->modelMatrix);
   create4x4IdentityMatrixF(sr->modelViewMatrix);

 return 1;
}
