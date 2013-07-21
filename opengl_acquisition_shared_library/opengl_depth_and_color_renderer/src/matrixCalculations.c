#include "matrixCalculations.h"
#include <stdio.h>
#include <math.h>


// Pre-calculated value of PI / 180.
#define kPI180   0.017453

// Pre-calculated value of 180 / PI.
#define k180PI  57.295780

// Converts degrees to radians.
#define degreesToRadians(x) (x * kPI180)

// Converts radians to degrees.
#define radiansToDegrees(x) (x * k180PI)

/*
inline double degreesToRadians(double degrees)
{
  return tan(0.25 * 3.141592653589793 );
}*/

void print3x3DMatrix(char * str , double * matrix4x4)
{
  fprintf( stderr, "  3x3 double %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf( stderr, "%f ",matrix4x4[0]);  fprintf( stderr, "%f ",matrix4x4[1]);  fprintf( stderr, "%f\n",matrix4x4[2]);
  fprintf( stderr, "%f ",matrix4x4[3]);  fprintf( stderr, "%f ",matrix4x4[4]);  fprintf( stderr, "%f\n",matrix4x4[5]);
  fprintf( stderr, "%f ",matrix4x4[6]);  fprintf( stderr, "%f ",matrix4x4[7]);  fprintf( stderr, "%f\n",matrix4x4[8]);
  fprintf( stderr, "--------------------------------------\n");
}


void print4x4DMatrix(char * str , double * matrix4x4)
{
  fprintf( stderr, " 4x4 double %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf( stderr, "  %f ",matrix4x4[0]);  fprintf( stderr, "%f ",matrix4x4[1]);  fprintf( stderr, "%f ",matrix4x4[2]);  fprintf( stderr, "%f\n",matrix4x4[3]);
  fprintf( stderr, "  %f ",matrix4x4[4]);  fprintf( stderr, "%f ",matrix4x4[5]);  fprintf( stderr, "%f ",matrix4x4[6]);  fprintf( stderr, "%f\n",matrix4x4[7]);
  fprintf( stderr, "  %f ",matrix4x4[8]);  fprintf( stderr, "%f ",matrix4x4[9]);  fprintf( stderr, "%f ",matrix4x4[10]); fprintf( stderr, "%f\n",matrix4x4[11]);
  fprintf( stderr, "  %f ",matrix4x4[12]); fprintf( stderr, "%f ",matrix4x4[13]); fprintf( stderr, "%f ",matrix4x4[14]); fprintf( stderr, "%f\n",matrix4x4[15]);
  fprintf( stderr, "--------------------------------------\n");
}


void copy4x4Matrix(double * out,double * in)
{
  out[0]=in[0];   out[1]=in[1];   out[2]=in[2];   out[3]=in[3];
  out[4]=in[4];   out[5]=in[5];   out[6]=in[6];   out[7]=in[7];
  out[8]=in[8];   out[9]=in[9];   out[10]=in[10]; out[11]=in[11];
  out[12]=in[12]; out[13]=in[13]; out[14]=in[14]; out[15]=in[15];
}


void matrix4x4Identity(double * m)
{
    //Diagonal
    m[0] = 1.0;  m[5] = 1.0;  m[10] = 1.0;  m[15] = 1.0;
    //Everything else is zero
    m[1] = 0.0; m[2] = 0.0;   m[3] = 0.0;   m[4] = 0.0;
    m[6] = 0.0; m[7] = 0.0;   m[8] = 0.0;   m[9] = 0.0;
    m[11] =0.0; m[12]= 0.0;   m[13]= 0.0;   m[14] = 0.0;
}

void matrix4x4Translate(double x, double y, double z, double * matrix)
{
    matrix4x4Identity(matrix);

    // Translate slots.
    matrix[12] = x;
    matrix[13] = y;
    matrix[14] = z;
}

void matrix4x4Scale(double sx, double sy, double sz, double * matrix)
{
    matrix4x4Identity(matrix);

    // Scale slots.
    matrix[0] = sx;
    matrix[5] = sy;
    matrix[10] = sz;
}

void matrix4x4RotateX(double degrees, double * matrix)
{
    double radians = degreesToRadians(degrees);

    matrix4x4Identity(matrix);

    // Rotate X formula.
    matrix[5] = cosf(radians);
    matrix[6] = -sinf(radians);
    matrix[9] = -matrix[6];
    matrix[10] = matrix[5];
}

void matrix4x4RotateY(double degrees, double * matrix)
{
    double radians = degreesToRadians(degrees);

    matrix4x4Identity(matrix);

    // Rotate Y formula.
    matrix[0] = cosf(radians);
    matrix[2] = sinf(radians);
    matrix[8] = -matrix[2];
    matrix[10] = matrix[0];
}

void matrixRotateZ(double degrees, double * matrix)
{
    double radians = degreesToRadians(degrees);

    matrix4x4Identity(matrix);

    // Rotate Z formula.
    matrix[0] = cosf(radians);
    matrix[1] = sinf(radians);
    matrix[4] = -matrix[1];
    matrix[5] = matrix[0];
}

int upscale3x3to4x4(double * mat3x3,double * mat4x4)
{
  if  ( (mat3x3==0)||(mat4x4==0) )   { return 0; }

  //TRANSPOSED RESULT
  mat4x4[0]=mat3x3[0]; mat4x4[1]=mat3x3[1]; mat4x4[2]=mat3x3[2];  mat4x4[3]=0.0;
  mat4x4[4]=mat3x3[3]; mat4x4[5]=mat3x3[4]; mat4x4[6]=mat3x3[5];  mat4x4[7]=0.0;
  mat4x4[8]=mat3x3[6]; mat4x4[9]=mat3x3[7]; mat4x4[10]=mat3x3[8]; mat4x4[11]=0.0;
  mat4x4[12]=0.0;      mat4x4[13]=0.0;      mat4x4[14]=0.0;       mat4x4[15]=1.0;

  return 1;
}


int convertRodriguezTo3x3(double * rodriguez , double * result)
{
  if ( (rodriguez==0) ||  (result==0) ) { return 0; }
  double x = rodriguez[0] , y = rodriguez[1] , z = rodriguez[2];
  double th = sqrt( x*x + y*y + z*z );
  double cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;

  /*
  //REAL RESULT
  result[0]=x*x * (1 - cosTh) + cosTh;        result[1]=x*y*(1 - cosTh) - z*sin(th);     result[2]=x*z*(1 - cosTh) + y*sin(th);
  result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;       result[5]=y*z*(1 - cosTh) - x*sin(th);
  result[6]=x*z*(1 - cosTh) - y*sin(th);        result[7]=y*z*(1 - cosTh) + x*sin(th);      result[8]=z*z*(1 - cosTh) + cosTh;
  */


  //  0 1 2    0 3 6
  //  3 4 5    1 4 7
  //  6 7 8    2 5 8

  //TRANSPOSED RESULT
  result[0]=x*x*(1 - cosTh) + cosTh;            result[3]=x*y*(1 - cosTh) - z*sin(th);     result[6]=x*z*(1 - cosTh) + y*sin(th);
  result[1]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;         result[7]=y*z*(1 - cosTh) - x*sin(th);
  result[2]=x*z*(1 - cosTh) - y*sin(th);        result[5]=y*z*(1 - cosTh) + x*sin(th);     result[8]=z*z*(1 - cosTh) + cosTh;


  fprintf(stderr,"rodriguez %0.2f %0.2f %0.2f\n ",rodriguez[0],rodriguez[1],rodriguez[2]);
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





int transpose4x4MatrixD(double * mat)
{
  if (mat==0) { return 0; }
  /*       -------  TRANSPOSE ------->
      0   1   2   3           0  4  8   12
      4   5   6   7           1  5  9   13
      8   9   10  11          2  6  10  14
      12  13  14  15          3  7  11  15   */
  double tmp;
  tmp = mat[1]; mat[1]=mat[4];  mat[4]=tmp;
  tmp = mat[2]; mat[2]=mat[8];  mat[8]=tmp;
  tmp = mat[3]; mat[3]=mat[12]; mat[12]=tmp;


  tmp = mat[6]; mat[6]=mat[9]; mat[9]=tmp;
  tmp = mat[13]; mat[13]=mat[7]; mat[7]=tmp;
  tmp = mat[14]; mat[14]=mat[11]; mat[11]=tmp;

  return 1;
}


int multiplyTwo4x4Matrices(double * result , double * matrixA , double * matrixB)
{
  if ( (matrixA==0) || (matrixB==0) || (result==0) ) { return 0; }

  fprintf(stderr,"Multiplying A and B \n");
  print4x4DMatrix("A", matrixA);
  print4x4DMatrix("B", matrixB);

  //MULTIPLICATION_RESULT FIRST ROW
  result[0]=matrixA[0] * matrixB[0] + matrixA[1] * matrixB[4]  + matrixA[2] * matrixB[8]  + matrixA[3] * matrixB[12];
  result[1]=matrixA[0] * matrixB[1] + matrixA[1] * matrixB[5]  + matrixA[2] * matrixB[9]  + matrixA[3] * matrixB[13];
  result[2]=matrixA[0] * matrixB[2] + matrixA[1] * matrixB[6]  + matrixA[2] * matrixB[10] + matrixA[3] * matrixB[14];
  result[3]=matrixA[0] * matrixB[3] + matrixA[1] * matrixB[7]  + matrixA[2] * matrixB[11] + matrixA[3] * matrixB[15];

  //MULTIPLICATION_RESULT SECOND ROW
  result[4]=matrixA[4] * matrixB[0] + matrixA[5] * matrixB[4]  + matrixA[6] * matrixB[8]  + matrixA[7] * matrixB[12];
  result[5]=matrixA[4] * matrixB[1] + matrixA[5] * matrixB[5]  + matrixA[6] * matrixB[9]  + matrixA[7] * matrixB[13];
  result[6]=matrixA[4] * matrixB[2] + matrixA[5] * matrixB[6]  + matrixA[6] * matrixB[10] + matrixA[7] * matrixB[14];
  result[7]=matrixA[4] * matrixB[3] + matrixA[5] * matrixB[7]  + matrixA[6] * matrixB[11] + matrixA[7] * matrixB[15];

  //MULTIPLICATION_RESULT FOURTH ROW
  result[8] =matrixA[8] * matrixB[0] + matrixA[9] * matrixB[4]  + matrixA[10] * matrixB[8]   + matrixA[11] * matrixB[12];
  result[9] =matrixA[8] * matrixB[1] + matrixA[9] * matrixB[5]  + matrixA[10] * matrixB[9]   + matrixA[11] * matrixB[13];
  result[10]=matrixA[8] * matrixB[2] + matrixA[9] * matrixB[6]  + matrixA[10] * matrixB[10]  + matrixA[11] * matrixB[14];
  result[11]=matrixA[8] * matrixB[3] + matrixA[9] * matrixB[7]  + matrixA[10] * matrixB[11]  + matrixA[11] * matrixB[15];

  result[12]=matrixA[12] * matrixB[0] + matrixA[13] * matrixB[4]  + matrixA[14] * matrixB[8]    + matrixA[15] * matrixB[12];
  result[13]=matrixA[12] * matrixB[1] + matrixA[13] * matrixB[5]  + matrixA[14] * matrixB[9]    + matrixA[15] * matrixB[13];
  result[14]=matrixA[12] * matrixB[2] + matrixA[13] * matrixB[6]  + matrixA[14] * matrixB[10]   + matrixA[15] * matrixB[14];
  result[15]=matrixA[12] * matrixB[3] + matrixA[13] * matrixB[7]  + matrixA[14] * matrixB[11]   + matrixA[15] * matrixB[15];

  print4x4DMatrix("AxB", result);

  return 1;
}
int multiplyVectorWith3x3Matrix(double * matrix, double * result)
{
  if ( (matrix==0) ||  (result==0) ) { return 0; }
  double x = matrix[0] , y = matrix[1] , z = matrix[2];
  double th = sqrt( x*x + y*y + z*z );
  double cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;

  /*
  //REAL RESULT
  result[0]=x*x * (1 - cosTh) + cosTh;        result[1]=x*y*(1 - cosTh) - z*sin(th);     result[2]=x*z*(1 - cosTh) + y*sin(th);
  result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;       result[5]=y*z*(1 - cosTh) - x*sin(th);
  result[6]=x*z*(1 - cosTh) - y*sin(th);        result[7]=y*z*(1 - cosTh) + x*sin(th);      result[8]=z*z*(1 - cosTh) + cosTh;
  */

  //  0 1 2    0 3 6
  //  3 4 5    1 4 7
  //  6 7 8    2 5 8

  //TRANSPOSED RESULT
  result[0]=x*x*(1 - cosTh) + cosTh;            result[3]=x*y*(1 - cosTh) - z*sin(th);     result[6]=x*z*(1 - cosTh) + y*sin(th);
  result[1]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;         result[7]=y*z*(1 - cosTh) - x*sin(th);
  result[2]=x*z*(1 - cosTh) - y*sin(th);        result[5]=y*z*(1 - cosTh) + x*sin(th);     result[8]=z*z*(1 - cosTh) + cosTh;

  return 1;
}


int convertRodriguezAndTransTo4x4(double * rodriguez , double * translation , double * matrix4x4 )
{
  double * matrix4x4Translation = (double * ) malloc ( sizeof(double) * 16 ); if (matrix4x4Translation==0) { return 0; }
  //double * matrix4x4Translation[16]={0}; <-- This produces w/e compiled code :P wtf
  matrix4x4Translate(translation[0],translation[1],translation[2],matrix4x4Translation);
  print4x4DMatrix("A Should be ", matrix4x4Translation);


  double * matrix4x4Rotation = (double * ) malloc ( sizeof(double) * 16 ); if (matrix4x4Rotation==0) { return 0; }
  double * matrix3x3Rotation = (double * ) malloc ( sizeof(double) * 9 );  if (matrix3x3Rotation==0) { return 0; }
  //double matrix4x4Rotation[16]={0};   <-- This produces w/e compiled code :P wtf
  //double matrix3x3Rotation[9]={0};
  convertRodriguezTo3x3(rodriguez,(double*) matrix3x3Rotation);
  upscale3x3to4x4((double*) matrix3x3Rotation,(double*)matrix4x4Rotation);
  print4x4DMatrix("B Should be ", matrix4x4Rotation);

  //Translate first rotate after

  multiplyTwo4x4Matrices((double*) matrix4x4, (double*) matrix4x4Translation , (double*) matrix4x4Rotation);

  //Append Translation
  //matrix4x4[3]=translation[0]; matrix4x4[7]=translation[1]; matrix4x4[11]=translation[2];

  //convertTranslationTo4x4(translation,matrix4x4);

  print4x4DMatrix("Rodriguez", matrix4x4);

  free(matrix4x4Translation);
  free(matrix4x4Rotation);
 return 1;
}




void InvertYandZAxisOpenGL4x4Matrix(double * result,double * matrix)
{
  fprintf(stderr,"Invert Y and Z axis\n");
  double modelView[16] = {1  ,  0  ,  0  , 0 ,
                          0  , -1  ,  0  , 0 ,
                          0  ,  0  , -1  , 0 ,
                          0  ,  0  ,  0  , 1 };

  multiplyTwo4x4Matrices(result,  modelView,matrix);
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
