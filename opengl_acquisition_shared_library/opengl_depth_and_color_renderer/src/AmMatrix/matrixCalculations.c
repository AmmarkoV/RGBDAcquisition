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
   result[0]=x*x * (1 - cosTh) + cosTh;          result[1]=x*y*(1 - cosTh) - z*sin(th);      result[2]=x*z*(1 - cosTh) + y*sin(th);
   result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;          result[5]=y*z*(1 - cosTh) - x*sin(th);
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


int convertRodriguezAndTranslationTo4x4DUnprojectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit)
{
  double * matrix3x3Rotation = alloc4x4Matrix();    if (matrix3x3Rotation==0) { return 0; }

  //Our translation vector is ready to be used!
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);

  //Our rodriguez vector should be first converted to a 3x3 Rotation matrix
  convertRodriguezTo3x3((double*) matrix3x3Rotation , rodriguez);

  //Shorthand variables for readable code :P
  double * m  = result4x4;
  double * rm = matrix3x3Rotation;
  double * tm = translation;


  //double scaleToDepthUnit = 1000.0; //Convert Unit to milimeters
  double Tx = tm[0]*scaleToDepthUnit;
  double Ty = tm[1]*scaleToDepthUnit;
  double Tz = tm[2]*scaleToDepthUnit;


  /*
      Here what we want to do is generate a 4x4 matrix that does the inverse transformation that our
      rodriguez and translation vector define

      In order to do that we should have the following be true

                                      (note the minus under)
      (   R  |  T  )       (   R trans |  - R trans * T  )         (   I  |  0   )
      (  --------- )    .  (  -------------------------- )     =   ( ----------- )
      (   0  |  1  )       (   0       |        1        )         (   0  |  I   )

      Using matlab to do the calculations we get the following matrix
  */

   m[0]=  rm[0];        m[1]= rm[3];        m[2]=  rm[6];       m[3]= -1.0 * ( rm[0]*Tx + rm[3]*Ty + rm[6]*Tz );
   m[4]=  rm[1];        m[5]= rm[4];        m[6]=  rm[7];       m[7]= -1.0 * ( rm[1]*Tx + rm[4]*Ty + rm[7]*Tz );
   m[8]=  rm[2];        m[9]= rm[5];        m[10]= rm[8];       m[11]=-1.0 * ( rm[2]*Tx + rm[5]*Ty + rm[8]*Tz );
   m[12]= 0.0;          m[13]= 0.0;         m[14]=0.0;          m[15]=1.0;


  print4x4DMatrix("ModelView", result4x4);
  free4x4Matrix(&matrix3x3Rotation);
  return 1;
}


int convertRodriguezAndTranslationToOpenGL4x4DProjectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit )
{
  double * matrix3x3Rotation = alloc4x4Matrix();    if (matrix3x3Rotation==0) { return 0; }

  //Our translation vector is ready to be used!
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);

  //Our rodriguez vector should be first converted to a 3x3 Rotation matrix
  convertRodriguezTo3x3((double*) matrix3x3Rotation , rodriguez);

  //Shorthand variables for readable code :P
  double * m  = result4x4;
  double * rm = matrix3x3Rotation;
  double * tm = translation;


  //double scaleToDepthUnit = 1000.0; //Convert Unit to milimeters
  double Tx = tm[0]*scaleToDepthUnit;
  double Ty = tm[1]*scaleToDepthUnit;
  double Tz = tm[2]*scaleToDepthUnit;

  /*
      Here what we want to do is generate a 4x4 matrix that does the normal transformation that our
      rodriguez and translation vector define
  */
   m[0]=  rm[0];        m[1]= rm[1];        m[2]=  rm[2];       m[3]= -Tx;
   m[4]=  rm[3];        m[5]= rm[4];        m[6]=  rm[5];       m[7]= -Ty;
   m[8]=  rm[6];        m[9]= rm[7];        m[10]= rm[8];       m[11]=-Tz;
   m[12]= 0.0;          m[13]= 0.0;         m[14]=0.0;          m[15]=1.0;

  print4x4DMatrix("ModelView", result4x4);

  fprintf(stderr,"Matrix will be transposed to become OpenGL format ( i.e. column major )\n");
  transpose4x4MatrixD(result4x4);

  free4x4Matrix(&matrix3x3Rotation);
  return 1;
}


int move3DPoint(double * resultPoint3D, double * transformation4x4, double * point3D  )
{
  return transform3DPointUsing4x4Matrix(resultPoint3D,transformation4x4,point3D);
}



void buildOpenGLProjectionForIntrinsicsAmmar   (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             double nearPlane,
                                             double farPlane
                                           )
{
   fprintf(stderr,"buildOpenGLProjectionForIntrinsics according to old Ammar code Image ( %u x %u )\n",imageWidth,imageHeight);
   fprintf(stderr,"fx %0.2f fy %0.2f , cx %0.2f , cy %0.2f , skew %0.2f \n",fx,fy,cx,cy,skew);
   fprintf(stderr,"Near %0.2f Far %0.2f \n",nearPlane,farPlane);


    // These parameters define the final viewport that is rendered into by
    // the camera.
    //     Left    Bottom   Right       Top
    double L = 0.0 , B = 0.0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    double N = nearPlane , F = farPlane;
    double R_sub_L = R-L , T_sub_B = T-B , F_sub_N = F-N , F_plus_N = F+N , F_mul_N = F*N;

    if  ( (R_sub_L==0) || (R_sub_L-1.0f==0) ||
          (T_sub_B==0) || (T_sub_B-1.0f==0) ||
          (F_sub_N==0) ) { fprintf(stderr,"Problem with image limigs R-L=%f , T-B=%f , F-N=%f\n",R_sub_L,T_sub_B,F_sub_N); }


   // set the viewport parameters
   viewport[0] = L; viewport[1] = B; viewport[2] = R_sub_L; viewport[3] = T_sub_B;

   //OpenGL Projection Matrix ready for loading ( column-major ) , also axis compensated
   frustum[0] = -2.0f*fx/R_sub_L;     frustum[1] = 0.0f;                 frustum[2] = 0.0f;                              frustum[3] = 0.0f;
   frustum[4] = 0.0f;                 frustum[5] = 2.0f*fy/T_sub_B;      frustum[6] = 0.0f;                              frustum[7] = 0.0f;
   frustum[8] = 2.0f*cx/R_sub_L-1.0f; frustum[9] = 2.0f*cy/T_sub_B-1.0f; frustum[10]=-1.0*(F_plus_N/F_sub_N);            frustum[11] = -1.0f;
   frustum[12]= 0.0f;                 frustum[13]= 0.0f;                 frustum[14]=-2.0f*F_mul_N/(F_sub_N);            frustum[15] = 0.0f;
   //Matrix already in OpenGL column major format
}




//Iasonas code
void buildOpenGLProjectionForIntrinsicsIasonas (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             double nearPlane,
                                             double farPlane
                                           )
{

   fprintf(stderr,"buildOpenGLProjectionForIntrinsics according to Iasonas Image ( %u x %u )\n",imageWidth,imageHeight);
   fprintf(stderr,"fx %0.2f fy %0.2f , cx %0.2f , cy %0.2f , skew %0.2f \n",fx,fy,cx,cy,skew);
   fprintf(stderr,"Near %0.2f Far %0.2f \n",nearPlane,farPlane);


    // These parameters define the final viewport that is rendered into by
    // the camera.
    //     Left    Bottom   Right       Top
    double L = 0 , B = 0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    double N = nearPlane , F = farPlane;
    double w = (double) imageWidth;
    double h = (double) imageHeight;

    double R_sub_L = R-L , T_sub_B = T-B , F_sub_N = F-N , /*F_plus_N = F+N,*/ F_mul_N = F*N;

    if  (R_sub_L==0) { fprintf(stderr,"R-L is negative (%0.2f-0) \n",R); }
    if  (T_sub_B==0) { fprintf(stderr,"T-B is negative (%0.2f-0) \n",T); }
    if  (F_sub_N==0) { fprintf(stderr,"F-N is negative (%0.2f-%0.2f) \n",F,N); }


   // set the viewport parameters
   viewport[0] = L;
   viewport[1] = B;
   viewport[2] = R_sub_L;
   viewport[3] = T_sub_B;

   frustum[0]=2.0 * (fx / w);     frustum[1]=0.0;             frustum[2]=2.0 * ( cx / w );                     frustum[3]=0.0;
   frustum[4]=0.0;                frustum[5]=-2.0 * (fy / h); frustum[6]=1 - ( 2.0 * ( cy / h ));              frustum[7]=0.0;
   frustum[8]=0.0;                frustum[9]=0.0;             frustum[10]=farPlane / ( F_sub_N );  frustum[11]=-1.0 * (F_mul_N / F_sub_N);
   frustum[12]=0.0;               frustum[13]=0.0;            frustum[14]=1.0;                                 frustum[15]=0.0;

   //Convert matrix to OpenGL  Column-major format
   transpose4x4MatrixD(frustum) ;
}





void buildOpenGLProjectionForIntrinsicsStrawlab   (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             unsigned int imageWidth,unsigned  int imageHeight,
                                             double nearPlane,
                                             double farPlane
                                           )
{

   float x0=0.0;
   float y0=0.0;

   fprintf(stderr,"buildOpenGLProjectionForIntrinsics according to http://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/ Image ( %u x %u )\n",imageWidth,imageHeight);
   fprintf(stderr,"fx %0.6f fy %0.6f , cx %0.6f , cy %0.6f , skew %0.6f \n",fx,fy,cx,cy,skew);
   fprintf(stderr,"Near %0.6f Far %0.6f \n",nearPlane,farPlane);
   fprintf(stderr,"x0  %0.6f y0 %0.6f \n",x0,y0);


    // These parameters define the final viewport that is rendered into by
    // the camera.
    //     Left    Bottom   Right       Top
    double L = 0.0 , B = 0.0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    double N = nearPlane , F = farPlane;
    double R_sub_L = R-L , T_sub_B = T-B , F_sub_N = F-N , F_plus_N = F+N , F_mul_N = F*N;

    if  ( (R_sub_L==0) || (R_sub_L-1.0f==0) ||
          (T_sub_B==0) || (T_sub_B-1.0f==0) ||
          (F_sub_N==0) ) { fprintf(stderr,"Problem with image limigs R-L=%f , T-B=%f , F-N=%f\n",R_sub_L,T_sub_B,F_sub_N); }


   // set the viewport parameters
   viewport[0] = L; viewport[1] = B; viewport[2] = R_sub_L; viewport[3] = T_sub_B;

/*
     http://strawlab.org/2011/11/05/augmented-reality-with-OpenGL/

     window_coords==’y up’ to be:

     [2*K00/width,  -2*K01/width,   (width - 2*K02 + 2*x0)/width,                            0]
     [          0, -2*K11/height, (height - 2*K12 + 2*y0)/height,                            0]
     [          0,             0, (-zfar - znear)/(zfar - znear), -2*zfar*znear/(zfar - znear)]
     [          0,             0,                             -1,                            0]


     With window_coords==’y down’, we have:

     [2*K00/width, -2*K01/width,    (width - 2*K02 + 2*x0)/width,                            0]
     [          0, 2*K11/height, (-height + 2*K12 + 2*y0)/height,                            0]
     [          0,            0,  (-zfar - znear)/(zfar - znear), -2*zfar*znear/(zfar - znear)]
     [          0,            0,                              -1,                            0]
*/


   //OpenGL Projection Matrix
   frustum[0] = 2* fx / R_sub_L;      frustum[1] = -2.0*skew/R_sub_L;    frustum[2] = (R_sub_L - 2.0*cx + 2.0*x0)/R_sub_L;              frustum[3] = 0.0f;
   frustum[4] = 0.0f;                 frustum[5] = 2.0*fy/T_sub_B;       frustum[6] = ((-1.0 * T_sub_B) + 2.0*cy + 2.0*y0)/T_sub_B ;    frustum[7] = 0.0f;
   frustum[8] = 0.0f;                 frustum[9] = 0.0f;                 frustum[10]= -1.0 * (F_plus_N/F_sub_N );                       frustum[11] = -2.0 * (F_mul_N/F_sub_N);
   frustum[12]= 0.0f;                 frustum[13]= 0.0f;                 frustum[14]= -1.0;                                             frustum[15] = 0.0f;

   //Convert matrix to OpenGL  Column-major format
   transpose4x4MatrixD(frustum) ;
}


void buildOpenGLProjectionForIntrinsicsMasteringOpenCV  (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             double nearPlane,
                                             double farPlane
                                           )
{

   float x0=0.0;
   float y0=0.0;

   fprintf(stderr,"buildOpenGLProjectionForIntrinsics according to Mastering OpenCV Image ( %u x %u )\n",imageWidth,imageHeight);
   fprintf(stderr,"fx %0.6f fy %0.6f , cx %0.6f , cy %0.6f , skew %0.6f \n",fx,fy,cx,cy,skew);
   fprintf(stderr,"Near %0.6f Far %0.6f \n",nearPlane,farPlane);
   fprintf(stderr,"x0  %0.6f y0 %0.6f \n",x0,y0);


    // These parameters define the final viewport that is rendered into by
    // the camera.
    //     Left    Bottom   Right       Top
    double L = 0.0 , B = 0.0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    double N = nearPlane , F = farPlane;
    double R_sub_L = R-L , T_sub_B = T-B , F_sub_N = F-N , F_plus_N = F+N , F_mul_N = F*N;


    if  ( (R_sub_L==0) || (R_sub_L-1.0f==0) ||
          (T_sub_B==0) || (T_sub_B-1.0f==0) ||
          (F_sub_N==0) ) { fprintf(stderr,"Problem with image limigs R-L=%f , T-B=%f , F-N=%f\n",R_sub_L,T_sub_B,F_sub_N); }


   // set the viewport parameters
   viewport[0] = L; viewport[1] = B; viewport[2] = R_sub_L; viewport[3] = T_sub_B;

   //OpenGL Projection Matrix ready for loading ( column-major ) , also axis compensated
   frustum[0] = -2.0 * fx / R_sub_L;      frustum[1] = 0.0f;                    frustum[2] = 0.0f;                        frustum[3] = 0.0f;
   frustum[4] = 0.0f;                     frustum[5] = 2.0*fy/T_sub_B;          frustum[6] = 0.0f;                        frustum[7] = 0.0f;
   frustum[8] = (2.0 * cx / R_sub_L)-1.0; frustum[9] = (2.0*cy / T_sub_B)-1.0;  frustum[10]= -1.0 * (F_plus_N/F_sub_N);   frustum[11] = -1.0;
   frustum[12]= 0.0f;                     frustum[13]= 0.0f;                    frustum[14]= -2.0 * (F_mul_N/F_sub_N);    frustum[15] = 0.0f;
   //Matrix already in OpenGL column major format
}





void buildOpenGLProjectionForIntrinsics (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             double nearPlane,
                                             double farPlane
                                           )
{
   //return buildOpenGLProjectionForIntrinsicsMasteringOpenCV( frustum, viewport , fx, fy, skew, cx,  cy, imageWidth, imageHeight, nearPlane, farPlane);
   //return buildOpenGLProjectionForIntrinsicsStrawlab( frustum, viewport , fx, fy, skew, cx,  cy, imageWidth, imageHeight, nearPlane, farPlane);
   //return buildOpenGLProjectionForIntrinsicsIasonas ( frustum, viewport , fx, fy, skew, cx,  cy, imageWidth, imageHeight, nearPlane, farPlane);
   return buildOpenGLProjectionForIntrinsicsAmmar ( frustum, viewport , fx, fy, skew, cx,  cy, imageWidth, imageHeight, nearPlane, farPlane);
}







void testMatrices()
{
  //testGJSolver();
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
