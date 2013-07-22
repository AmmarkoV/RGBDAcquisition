#include "matrixCalculations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix3x3Tools.h"
#include "matrix4x4Tools.h"





int convertRodriguezTo3x3(double * matrix, double * result)
{
  if ( (matrix==0) ||  (result==0) ) { return 0; }
  double x = matrix[0] , y = matrix[1] , z = matrix[2];
  double th = sqrt( x*x + y*y + z*z );
  double cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;


  //Switch to control what kind of a result to give :P
  #define PRODUCE_TRANSPOSED_RESULT 0
  // REGULAR  TRANSPOSED
  //  0 1 2     0 3 6
  //  3 4 5     1 4 7
  //  6 7 8     2 5 8

  #if PRODUCE_TRANSPOSED_RESULT
    //TRANSPOSED RESULT
    result[0]=x*x*(1 - cosTh) + cosTh;            result[3]=x*y*(1 - cosTh) - z*sin(th);     result[6]=x*z*(1 - cosTh) + y*sin(th);
    result[1]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;         result[7]=y*z*(1 - cosTh) - x*sin(th);
    result[2]=x*z*(1 - cosTh) - y*sin(th);        result[5]=y*z*(1 - cosTh) + x*sin(th);     result[8]=z*z*(1 - cosTh) + cosTh;
  #else
   //NORMAL RESULT
   result[0]=x*x * (1 - cosTh) + cosTh;          result[1]=x*y*(1 - cosTh) - z*sin(th);     result[2]=x*z*(1 - cosTh) + y*sin(th);
   result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;       result[5]=y*z*(1 - cosTh) - x*sin(th);
   result[6]=x*z*(1 - cosTh) - y*sin(th);        result[7]=y*z*(1 - cosTh) + x*sin(th);      result[8]=z*z*(1 - cosTh) + cosTh;
  #endif

  fprintf(stderr,"rodriguez %0.2f %0.2f %0.2f\n ",matrix[0],matrix[1],matrix[2]);
  print3x3DMatrix("Rodriguez Initial", result);

  return 1;
}



int convertTranslationTo4x4(double * translation, double * result)
{
  if ( (translation==0) ||  (result==0) ) { return 0; }
  double x = translation[0] , y = translation[1] , z = translation[2];

  result[0]=1.0; result[1]=0;   result[2]=0;    result[3]=x;
  result[4]=0;   result[5]=1.0; result[6]=0;    result[7]=y;
  result[8]=0;   result[9]=0;   result[10]=1.0; result[11]=z;
  result[12]=0;  result[13]=0;  result[14]=0;   result[15]=1.0;

  return 1;
}



void InvertYandZAxisOpenGL4x4Matrix(double * result,double * matrix)
{
  fprintf(stderr,"Invert Y and Z axis\n");
  double * invertOp = (double * ) malloc ( sizeof(double) * 16 );
  if (invertOp==0) { return; }

  create4x4IdentityMatrix(invertOp);
  invertOp[5]=-1;   invertOp[10]=-1;
  multiplyTwo4x4Matrices(result, matrix, invertOp);
  free(invertOp);
}


int convertRodriguezAndTransTo4x4(double * rodriguez , double * translation , double * matrix4x4 )
{
  double * matrix4x4Rotation = (double * ) malloc ( sizeof(double) * 16 ); if (matrix4x4Rotation==0) { return 0; }
  double * matrix3x3Rotation = (double * ) malloc ( sizeof(double) * 9 );  if (matrix3x3Rotation==0) { return 0; }

  fprintf(stderr,"translation %0.2f %0.2f %0.2f\n ",translation[0],translation[1],translation[2]);
  convertRodriguezTo3x3(rodriguez,(double*) matrix3x3Rotation);

  double * rm = matrix3x3Rotation;
  double * tm = translation;

  //Compose a 4x4 matrix with the translation and rotation , Normal ,
  matrix4x4[0]= rm[0];    matrix4x4[1]=rm[1];      matrix4x4[2]=rm[2];      matrix4x4[3]=tm[0];
  matrix4x4[4]= rm[3];    matrix4x4[5]=rm[4];      matrix4x4[6]=rm[5];      matrix4x4[7]=-tm[1];
  matrix4x4[8]= rm[6];    matrix4x4[9]=rm[7];      matrix4x4[10]=rm[8];     matrix4x4[11]=-tm[2];
  matrix4x4[12]= 0.0;     matrix4x4[13]=0.0;       matrix4x4[14]=0.0;       matrix4x4[15]=1.0;
  print4x4DMatrix("Rodriguez ModelView Result", matrix4x4);

  //Compose a 4x4 matrix with the translation and rotation , OPENGL row /column major setting
  matrix4x4[0]= rm[0];    matrix4x4[1]=rm[3];      matrix4x4[2]=rm[6];      matrix4x4[3]=0.0;
  matrix4x4[4]= rm[1];    matrix4x4[5]=rm[4];      matrix4x4[6]=rm[7];      matrix4x4[7]=0.0;
  matrix4x4[8]= rm[2];    matrix4x4[9]=rm[5];      matrix4x4[10]=rm[8];     matrix4x4[11]=0.0;
  matrix4x4[12]=tm[0];    matrix4x4[13]=-tm[1];    matrix4x4[14]=-tm[2];    matrix4x4[15]=1.0;
  print4x4DMatrix("Rodriguez ModelView Result OpenGL", matrix4x4);

  free(matrix4x4Rotation);
  free(matrix3x3Rotation);

 return 1;
}






void buildOpenGLProjectionForIntrinsics   (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             int imageWidth, int imageHeight,
                                             double nearPlane,
                                             double farPlane
                                           )
{
   fprintf(stderr,"buildOpenGLProjectionForIntrinsics Image ( %u x %u )\n",imageWidth,imageHeight);
   fprintf(stderr,"fx %0.2f fy %0.2f , cx %0.2f , cy %0.2f , skew %0.2f \n",fx,fy,cx,cy,skew);
   fprintf(stderr,"Near %0.2f Far %0.2f \n",nearPlane,farPlane);


    // These parameters define the final viewport that is rendered into by
    // the camera.
    //     Left    Bottom   Right       Top
    double L = 0 , B = 0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    double N = nearPlane , F = farPlane;


    double R_sub_L = R-L;
    double T_sub_B = T-B;
    double F_sub_N = F-N;

    if  (R_sub_L==0) { fprintf(stderr,"R-L is negative (%0.2f-0) \n",R); }
    if  (T_sub_B==0) { fprintf(stderr,"T-B is negative (%0.2f-0) \n",T); }
    if  (F_sub_N==0) { fprintf(stderr,"F-N is negative (%0.2f-%0.2f) \n",F,N); }


    // set the viewport parameters
    viewport[0] = L;
    viewport[1] = B;
    viewport[2] = R_sub_L;
    viewport[3] = T_sub_B;


 frustum[0]=2.0 * fx / imageWidth; frustum[1]=0.0;                    frustum[2]=2.0 * ( cx / imageWidth ) - 1.0;                     frustum[3]=0.0;
 frustum[4]=0.0;                   frustum[5]=2.0 * fy / imageHeight; frustum[6]=2.0 * ( cy / imageHeight ) - 1.0;                    frustum[7]=0.0;
 frustum[8]=0.0;                   frustum[9]=0.0;                    frustum[10]=-( farPlane+nearPlane ) / ( farPlane - nearPlane ); frustum[11]=-2.0 * farPlane * nearPlane / ( farPlane - nearPlane );
 frustum[12]=0.0;                  frustum[13]=0.0;                   frustum[14]=-1.0;                                               frustum[15]=0.0;

}












void testMatrices()
{
  return ;
  double A[16]={ 1 ,2 ,3 ,4,
                 5 ,6 ,7 ,8,
                 9 ,10,11,12,
                 13,14,15,16
                };


  double B[16]={ 1 ,2 ,3 ,4,
                 4 ,3 ,2 ,1,
                 1 ,2 ,3 ,4,
                 4 ,3 ,2 ,1
                };

  double Res[16]={0};

  multiplyTwo4x4Matrices(Res,A,B);
/*
  28.000000 26.000000 24.000000 22.000000
  68.000000 66.000000 64.000000 62.000000
  108.000000 106.000000 104.000000 102.000000
  148.000000 146.000000 144.000000 142.000000*/

}
