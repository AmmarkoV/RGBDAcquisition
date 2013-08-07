#include "matrixCalculations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matrix3x3Tools.h"
#include "matrix4x4Tools.h"
#include "solveLinearSystemGJ.h"

int convertRodriguezTo3x3(double * result,double * matrix)
{
  if ( (matrix==0) ||  (result==0) ) { return 0; }


  double x = matrix[0] , y = matrix[1] , z = matrix[2];
  double th = sqrt( x*x + y*y + z*z );
  double cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;

  if ( th < 0.00001 )
    {
       create3x3IdentityMatrix(result);
       return 1;
    }

   //NORMAL RESULT
   result[0]=x*x * (1 - cosTh) + cosTh;          result[1]=x*y*(1 - cosTh) - z*sin(th);     result[2]=x*z*(1 - cosTh) + y*sin(th);
   result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;       result[5]=y*z*(1 - cosTh) - x*sin(th);
   result[6]=x*z*(1 - cosTh) - y*sin(th);        result[7]=y*z*(1 - cosTh) + x*sin(th);      result[8]=z*z*(1 - cosTh) + cosTh;

  fprintf(stderr,"rodriguez %f %f %f\n ",matrix[0],matrix[1],matrix[2]);
  print3x3DMatrix("Rodriguez Initial", result);

  return 1;
}


void changeYandZAxisOpenGL4x4Matrix(double * result,double * matrix)
{
  fprintf(stderr,"Invert Y and Z axis\n");
  double * invertOp = (double * ) malloc ( sizeof(double) * 16 );
  if (invertOp==0) { return; }

  create4x4IdentityMatrix(invertOp);
  invertOp[5]=-1;   invertOp[10]=-1;
  multiplyTwo4x4Matrices(result, matrix, invertOp);
  free(invertOp);
}

int projectPointsFrom3Dto2D(double * x2D, double * y2D , double * x3D, double *y3D , double * z3D , double * intrinsics , double * rotation3x3 , double * translation)
{
  double fx = intrinsics[0];
  double fy = intrinsics[4];
  double cx = intrinsics[2];
  double cy = intrinsics[5];

  double * t = translation;
  double * r = rotation3x3;

  //Result
  //fx * t0 + cx * t2 + (x3D) * ( fx * r0 + cx * r6 )  + (y3D) * ( fx * r1 + cx * r7 ) + (z3D) * (fx * r2 +cx * r8) / t3 + r7 x3D + r8 * y3D + r9 * z3D
  //fy * t1 + cy * t2 + x3D * ( fy * r3 + cy * r6 )  + y3D * ( fy * r4 + cy * r7 ) + z3D * (fy * r5 +cy * r8) / t3 + r7 x3D + r8 * y3D + r9 * z3D
  //1

  double x2DBuf =  fx * t[0] + cx * t[2] + (*x3D) * ( fx * r[0] + cx * r[6] )  + (*y3D) * ( fx * r[1] + cx * r[7] ) + (*z3D) * (fx * r[2] +cx * r[8]);
  double y2DBuf =  fy * t[1] + cy * t[2] + (*x3D) * ( fy * r[3] + cy * r[6] )  + (*y3D) * ( fy * r[4] + cy * r[7] ) + (*z3D) * (fy * r[5] +cy * r[8]);
  double scale =   t[2] + r[6] * (*x3D) + r[7] * (*y3D) + r[8] * (*z3D);

  if ( scale == 0.0 ) { fprintf(stderr,"could not projectPointsFrom3Dto2D"); return 0; }
  *x2D = x2DBuf / scale;
  *y2D = y2DBuf / scale;

 return 1;
}

int convertRodriguezAndTranslationToOpenGL4x4DMatrix(double * result4x4, double * rodriguez , double * translation  )
{
  double scale = 1.0;
  double * matrix3x3Rotation = alloc4x4Matrix();    if (matrix3x3Rotation==0) { return 0; }

  //Our translation vector is ready to be used!
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);

  //Our rodriguez vector should be first converted to a 3x3 Rotation matrix
  convertRodriguezTo3x3((double*) matrix3x3Rotation , rodriguez);

  //Shorthand variables for readable code :P
  double * m  = result4x4;
  double * rm = matrix3x3Rotation;
  double * tm = translation;

  //All steps at one step!
  // rm has been transposed so that we have the inverse rotation ( based on camera coordinates )
  // Thats why we go 0->3->6 , 1->4->7 , 2->5->8
  // translations have a minus so that they also go inverse
  // The result matrix is transposed so that OpenGL can read it correctly ( because OpenGL uses column major matrices )
   m[0]=  rm[0];        m[1]= rm[3];        m[2]=  rm[6];       m[3]= 0.0;
   m[4]=  rm[1];        m[5]= rm[4];        m[6]=  rm[7];       m[7]= 0.0;
   m[8]=  rm[2];        m[9]= rm[5];        m[10]= rm[8];       m[11]=0.0;
   m[12]=-tm[0]*scale;  m[13]=-tm[1]*scale; m[14]=-tm[2]*scale; m[15]=1.0;

  print4x4DMatrix("ModelView Theory Inverted", result4x4);
  free4x4Matrix(&matrix3x3Rotation);
  return 1;
}



int move3DPoint(double * resultPoint3D, double * transformation4x4, double * point3D  )
{
  return transform3DPointUsing4x4Matrix(resultPoint3D,transformation4x4,point3D);
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
    double L = 0.0 , B = 0.0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    double N = nearPlane , F = farPlane;
    double R_sub_L = R-L , T_sub_B = T-B , F_sub_N = F-N;

    if  ( (R_sub_L==0) || (R_sub_L-1.0f==0) ||
          (T_sub_B==0) || (T_sub_B-1.0f==0) ||
          (F_sub_N==0) ) { fprintf(stderr,"Problem with image limigs R-L=%f , T-B=%f , F-N=%f\n",R_sub_L,T_sub_B,F_sub_N); }


   // set the viewport parameters
   viewport[0] = L; viewport[1] = B; viewport[2] = R_sub_L; viewport[3] = T_sub_B;

   //OpenGL Projection Matrix ready for loading ( column-major ) , also axis compensated
   frustum[0] = -2.0f*fx/R_sub_L;     frustum[1] = 0.0f;                 frustum[2] = 0.0f;                              frustum[3] = 0.0f;
   frustum[4] = 0.0f;                 frustum[5] = 2.0f*fy/T_sub_B;      frustum[6] = 0.0f;                              frustum[7] = 0.0f;
   frustum[8] = 2.0f*cx/R_sub_L-1.0f; frustum[9] = 2.0f*cy/T_sub_B-1.0f; frustum[10]=-(farPlane+nearPlane)/(F_sub_N);    frustum[11] = -1.0f;
   frustum[12]= 0.0f;                 frustum[13]= 0.0f;                 frustum[14]=-2.0f*farPlane*nearPlane/(F_sub_N); frustum[15] = 0.0f;
}











void testMatrices()
{
  testGJSolver();
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
