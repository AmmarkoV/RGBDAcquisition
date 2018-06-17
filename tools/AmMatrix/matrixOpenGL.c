#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "matrixOpenGL.h"

#include "matrix4x4Tools.h"


int convertRodriguezTo3x3(double * result,double * matrix)
{
  if ( (matrix==0) ||  (result==0) ) { return 0; }


  double x = matrix[0] , y = matrix[1] , z = matrix[2];
  double th = sqrt( x*x + y*y + z*z );
  double cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;

  if ( th < 0.00001 )
    {
       result[0]=1.0f;  result[1]=0.0f; result[2]=0.0f;
       result[3]=0.0f;  result[4]=1.0f; result[5]=0.0f;
       result[6]=0.0f;  result[7]=0.0f; result[8]=1.0f;

       //create3x3IdentityMatrix(result);
       return 1;
    }

   //NORMAL RESULT
   result[0]=x*x * (1 - cosTh) + cosTh;          result[1]=x*y*(1 - cosTh) - z*sin(th);      result[2]=x*z*(1 - cosTh) + y*sin(th);
   result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;          result[5]=y*z*(1 - cosTh) - x*sin(th);
   result[6]=x*z*(1 - cosTh) - y*sin(th);        result[7]=y*z*(1 - cosTh) + x*sin(th);      result[8]=z*z*(1 - cosTh) + cosTh;

  #if PRINT_MATRIX_DEBUGGING
   fprintf(stderr,"rodriguez %f %f %f\n ",matrix[0],matrix[1],matrix[2]);
   print3x3DMatrix("Rodriguez Initial", result);
  #endif // PRINT_MATRIX_DEBUGGING

  return 1;
}


void changeYandZAxisOpenGL4x4Matrix(double * result,double * matrix)
{
  #if PRINT_MATRIX_DEBUGGING
   fprintf(stderr,"Invert Y and Z axis\n");
  #endif // PRINT_MATRIX_DEBUGGING

  double * invertOp = (double * ) malloc ( sizeof(double) * 16 );
  if (invertOp==0) { return; }

  create4x4IdentityMatrix(invertOp);
  invertOp[5]=-1;   invertOp[10]=-1;
  multiplyTwo4x4Matrices(result, matrix, invertOp);
  free(invertOp);
}

int convertRodriguezAndTranslationTo4x4DUnprojectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit)
{
  double * matrix3x3Rotation = alloc4x4Matrix();    if (matrix3x3Rotation==0) { return 0; }

  //Our translation vector is ready to be used!
  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);
  #endif // PRINT_MATRIX_DEBUGGING

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
  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);
  #endif // PRINT_MATRIX_DEBUGGING


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


  #if PRINT_MATRIX_DEBUGGING
   print4x4DMatrix("ModelView", result4x4);
   fprintf(stderr,"Matrix will be transposed to become OpenGL format ( i.e. column major )\n");
  #endif // PRINT_MATRIX_DEBUGGING

  transpose4x4MatrixD(result4x4);

  free4x4Matrix(&matrix3x3Rotation);
  return 1;
}



void buildOpenGLProjectionForIntrinsics   (
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
          (F_sub_N==0) ) { fprintf(stderr,"Problem with image limits R-L=%f , T-B=%f , F-N=%f\n",R_sub_L,T_sub_B,F_sub_N); }


   // set the viewport parameters
   viewport[0] = L; viewport[1] = B; viewport[2] = R_sub_L; viewport[3] = T_sub_B;

   //OpenGL Projection Matrix ready for loading ( column-major ) , also axis compensated
   frustum[0] = -2.0f*fx/R_sub_L;     frustum[1] = 0.0f;                 frustum[2] = 0.0f;                              frustum[3] = 0.0f;
   frustum[4] = 0.0f;                 frustum[5] = 2.0f*fy/T_sub_B;      frustum[6] = 0.0f;                              frustum[7] = 0.0f;
   frustum[8] = 2.0f*cx/R_sub_L-1.0f; frustum[9] = 2.0f*cy/T_sub_B-1.0f; frustum[10]=-1.0*(F_plus_N/F_sub_N);            frustum[11] = -1.0f;
   frustum[12]= 0.0f;                 frustum[13]= 0.0f;                 frustum[14]=-2.0f*F_mul_N/(F_sub_N);            frustum[15] = 0.0f;
   //Matrix already in OpenGL column major format



   //TROUBLESHOOTING Left To Right Hand conventions , Thanks Damien 24-06-15
   double identMat[16];
   double finalFrutstrum[16];
   create4x4IdentityMatrix(identMat);
   identMat[10]=-1;
   multiplyTwo4x4Matrices(finalFrutstrum,identMat,frustum);
   copy4x4Matrix(frustum,finalFrutstrum);

}



//matrix will receive the calculated perspective matrix.
//You would have to upload to your shader
// or use glLoadMatrixf if you aren't using shaders.
void glhFrustumf2(
                  float *matrix,
                  float left,
                  float right,
                  float bottom,
                  float top,
                  float znear,
                  float zfar
                 )
{
    float temp, temp2, temp3, temp4;
    temp = 2.0 * znear;
    temp2 = right - left;
    temp3 = top - bottom;
    temp4 = zfar - znear;
    matrix[0] = temp / temp2;
    matrix[1] = 0.0;
    matrix[2] = 0.0;
    matrix[3] = 0.0;
    matrix[4] = 0.0;
    matrix[5] = temp / temp3;
    matrix[6] = 0.0;
    matrix[7] = 0.0;
    matrix[8] = (right + left) / temp2;
    matrix[9] = (top + bottom) / temp3;
    matrix[10] = (-zfar - znear) / temp4;
    matrix[11] = -1.0;
    matrix[12] = 0.0;
    matrix[13] = 0.0;
    matrix[14] = (-temp * zfar) / temp4;
    matrix[15] = 0.0;
}


void glhPerspectivef2(
                      float *matrix,
                      float fovyInDegrees,
                      float aspectRatioV,
                      float znear,
                      float zfar
                     )
{
    float ymax, xmax;
    ymax = znear * tan(fovyInDegrees * M_PI / 360.0);
    //ymin = -ymax;
    //xmin = -ymax * aspectRatioV;
    xmax = ymax * aspectRatioV;
    glhFrustumf2(matrix, -xmax, xmax, -ymax, ymax, znear, zfar);
}




void gldPerspective(
                     double *matrix,
                     double fovx,
                     double aspect,
                     double zNear,
                     double zFar
                   )
{
   // This code is based off the MESA source for gluPerspective
   // *NOTE* This assumes GL_PROJECTION is the current matrix
   double xmin, xmax, ymin, ymax;

   xmax = zNear * tan(fovx * M_PI / 360.0);
   xmin = -xmax;

   ymin = xmin / aspect;
   ymax = xmax / aspect;

   // Set up the projection matrix
   matrix[0] = (2.0 * zNear) / (xmax - xmin);
   matrix[1] = 0.0;
   matrix[2] = 0.0;
   matrix[3] = 0.0;
   matrix[4] = 0.0;
   matrix[5] = (2.0 * zNear) / (ymax - ymin);
   matrix[6] = 0.0;
   matrix[7] = 0.0;
   matrix[8] = (xmax + xmin) / (xmax - xmin);
   matrix[9] = (ymax + ymin) / (ymax - ymin);
   matrix[10] = -(zFar + zNear) / (zFar - zNear);
   matrix[11] = -1.0;
   matrix[12] = 0.0;
   matrix[13] = 0.0;
   matrix[14] = -(2.0 * zFar * zNear) / (zFar - zNear);
   matrix[15] = 0.0;

   // Add to current matrix
   //glMultMatrixd(matrix);
}



void lookAt(
             float * matrix ,
             float eyex, float eyey, float eyez,
	         float centerx, float centery, float centerz,
	         float upx, float upy, float upz
	        )
{
   float x[3], y[3], z[3];
   float mag;

   /* Make rotation matrix */

   /* Z vector */
   z[0] = eyex - centerx;
   z[1] = eyey - centery;
   z[2] = eyez - centerz;
   mag = sqrt( z[0] * z[0] + z[1] * z[1] + z[2] * z[2]);
   if (mag)
    {
      z[0] /= mag;
      z[1] /= mag;
      z[2] /= mag;
    }

   /* Y vector */
   y[0] = upx;
   y[1] = upy;
   y[2] = upz;

   /* X vector = Y cross Z */
   x[0] = y[1] * z[2] - y[2] * z[1];
   x[1] = -y[0] * z[2] + y[2] * z[0];
   x[2] = y[0] * z[1] - y[1] * z[0];

   /* Recompute Y = Z cross X */
   y[0] = z[1] * x[2] - z[2] * x[1];
   y[1] = -z[0] * x[2] + z[2] * x[0];
   y[2] = z[0] * x[1] - z[1] * x[0];

   /* mpichler, 19950515 */
   /* cross product gives area of parallelogram, which is < 1.0 for
    * non-perpendicular unit-length vectors; so normalize x, y here
    */

   mag = sqrt(  x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
   if (mag)
    {
      x[0] /= mag;
      x[1] /= mag;
      x[2] /= mag;
    }

   mag = sqrt( y[0] * y[0] + y[1] * y[1] + y[2] * y[2]);
   if (mag)
    {
      y[0] /= mag;
      y[1] /= mag;
      y[2] /= mag;
    }

   float initial[16];
   initial[0] = x[0]; initial[1] = x[1]; initial[2] = x[2]; initial[3] = 0.0;
   initial[4] = y[0]; initial[5] = y[1]; initial[6] = y[2]; initial[7] = 0.0;
   initial[8] = z[0]; initial[9] = z[1]; initial[10]= z[2]; initial[11]= 0.0;
   initial[12]= 0.0;  initial[13]= 0.0;  initial[14]= 0.0;  initial[15]= 1.0;


   /* Translate Eye to Origin */
   //glTranslatef(-eyex, -eyey, -eyez);
   float translation[16];
   create4x4FTranslationMatrix(translation , -eyex, -eyey, -eyez );
   multiplyTwo4x4FMatrices(matrix , initial , translation);

}
