#include "simpleRenderer.h"
#include "matrix4x4Tools.h"
#include "matrixOpenGL.h"
#include <stdio.h>



void create4x4ModelTransformationWithNonCenteredCoordinates(
                                                            double * m ,
                                                            //Rotation Component
                                                            double rotationX,//heading
                                                            double rotationY,//pitch
                                                            double rotationZ,//roll
                                                            unsigned int rotationOrder,
                                                            //Translation Component
                                                            double x, double y, double z ,
                                                            double centerX, double centerY, double centerZ ,
                                                            double scaleX, double scaleY, double scaleZ
                                                           )
{
   if (m==0) {return;}

    //fprintf(stderr,"Asked for a model transformation with RPY(%0.2f,%0.2f,%0.2f)",rollInDegrees,pitchInDegrees,yawInDegrees);
    //fprintf(stderr,"XYZ(%0.2f,%0.2f,%0.2f)",x,y,z);
    //fprintf(stderr,"scaled(%0.2f,%0.2f,%0.2f)\n",scaleX,scaleY,scaleZ);


    double translationToCenter[16];
    create4x4TranslationMatrix(
                               translationToCenter,
                               -centerX,
                               -centerY,
                               -centerZ
                              );

    double intermediateMatrixTranslation[16];
    create4x4TranslationMatrix(
                               intermediateMatrixTranslation,
                               x,
                               y,
                               z
                              );


    double intermediateMatrixRotation[16];
    if (rotationOrder==ROTATION_ORDER_RPY)
    {
     //This is the old way to do this rotation
     doRPYTransformation(
                         intermediateMatrixRotation,
                         rotationZ,//roll,
                         rotationY,//pitch
                         rotationX//heading
                        );
    } else
    {
     //fprintf(stderr,"Using new model transform code\n");
     create4x4MatrixFromEulerAnglesWithRotationOrder(
                                                     intermediateMatrixRotation ,
                                                     rotationX,
                                                     rotationY,
                                                     rotationZ,
                                                     rotationOrder
                                                    );
    }


  if ( (scaleX!=1.0) || (scaleY!=1.0) || (scaleZ!=1.0) )
      {
        double intermediateScalingMatrix[16];
        create4x4ScalingMatrix(intermediateScalingMatrix,scaleX,scaleY,scaleZ);
        multiplyFour4x4Matrices(m,intermediateMatrixTranslation,intermediateMatrixRotation,translationToCenter,intermediateScalingMatrix);
      } else
      {
        // multiplyTwo4x4Matrices(m,intermediateMatrixTranslation,intermediateMatrixRotation);
         multiplyThree4x4Matrices(m,intermediateMatrixTranslation,intermediateMatrixRotation,translationToCenter);
      }
}













int simpleRendererRender(
                         struct simpleRenderer * sr ,
                         float * position3D,
                         float * center3D,
                         float * output2DX,
                         float * output2DY
                        )
{
 double modelTransformationD[16];
 float  modelTransformationF[16];

 float ourPosition3DCopy[4];
 ourPosition3DCopy[0]=position3D[0]+sr->objectOffsetPosition[0];
 ourPosition3DCopy[1]=position3D[1]+sr->objectOffsetPosition[1];
 ourPosition3DCopy[2]=position3D[2]+sr->objectOffsetPosition[2];
 ourPosition3DCopy[3]=position3D[3];


 double position3DX = (double) ourPosition3DCopy[0];
 double position3DY = (double) ourPosition3DCopy[1];
 double position3DZ = (double) ourPosition3DCopy[2];
 ///--------------------------------------------------------------------
 create4x4ModelTransformationWithNonCenteredCoordinates(
                                                        modelTransformationD,
                                                        //Rotation Component
                                                        (double) sr->objectOffsetRotation[0],//heading,
                                                        (double) sr->objectOffsetRotation[1],//pitch,
                                                        (double) sr->objectOffsetRotation[2],//roll,
                                                        ROTATION_ORDER_RPY,
                                                        //Translation Component
                                                        (double) sr->objectOffsetPosition[0],
                                                        (double) sr->objectOffsetPosition[1],
                                                        (double) sr->objectOffsetPosition[2],
                                                        //Center Of Coordinate System to rotate @
                                                        (double) center3D[0],
                                                        (double) center3D[1],
                                                        (double) center3D[2],
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

 //Force rendering to be in the center .. :P
 //ourPosition3DCopy[0]-=center3D[0];
 //ourPosition3DCopy[1]-=center3D[1];
 //ourPosition3DCopy[2]-=center3D[2];

  _glhProjectf(
                ourPosition3DCopy,
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

   double viewMatrixD[16];
   create4x4ScalingMatrix(viewMatrixD,-1.0,1.0,1.0);
   copy4x4DMatrixToF(sr->viewMatrix,viewMatrixD);

   //Initialization of matrices not yet used
   create4x4IdentityMatrixF(sr->modelMatrix);
   create4x4IdentityMatrixF(sr->modelViewMatrix);

 return 1;
}
