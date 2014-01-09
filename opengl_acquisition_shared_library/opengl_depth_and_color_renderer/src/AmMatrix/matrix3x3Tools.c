#include "matrix3x3Tools.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

enum mat3x3Item
{
    I11 = 0 , I12 , I13 ,
    I21     , I22 , I23 ,
    I31     , I32 , I33
};

double * alloc3x3Matrix()
{
  return malloc ( sizeof(double) * 16 );
}

void free3x3Matrix(double ** mat)
{
  if (mat==0) { return ; }
  if (*mat==0) { return ; }
  free(*mat);
  *mat=0;
}


void print3x3FMatrix(char * str , float * matrix4x4)
{
  fprintf( stderr, "  3x3 float %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf( stderr, "%f ",matrix4x4[0]);  fprintf( stderr, "%f ",matrix4x4[1]);  fprintf( stderr, "%f\n",matrix4x4[2]);
  fprintf( stderr, "%f ",matrix4x4[3]);  fprintf( stderr, "%f ",matrix4x4[4]);  fprintf( stderr, "%f\n",matrix4x4[5]);
  fprintf( stderr, "%f ",matrix4x4[6]);  fprintf( stderr, "%f ",matrix4x4[7]);  fprintf( stderr, "%f\n",matrix4x4[8]);
  fprintf( stderr, "--------------------------------------\n");
}

void print3x3DMatrix(char * str , double * matrix4x4)
{
  fprintf( stderr, "  3x3 double %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf( stderr, "%f ",matrix4x4[0]);  fprintf( stderr, "%f ",matrix4x4[1]);  fprintf( stderr, "%f\n",matrix4x4[2]);
  fprintf( stderr, "%f ",matrix4x4[3]);  fprintf( stderr, "%f ",matrix4x4[4]);  fprintf( stderr, "%f\n",matrix4x4[5]);
  fprintf( stderr, "%f ",matrix4x4[6]);  fprintf( stderr, "%f ",matrix4x4[7]);  fprintf( stderr, "%f\n",matrix4x4[8]);
  fprintf( stderr, "--------------------------------------\n");
}


void print3x3DScilabMatrix(char * str , double * matrix4x4)
{
  fprintf( stderr, "%s = [ %f ",str,matrix4x4[0]);  fprintf( stderr, "%f ",matrix4x4[1]);  fprintf( stderr, "%f ; ",matrix4x4[2]);
  fprintf( stderr, "%f ",matrix4x4[3]);  fprintf( stderr, "%f ",matrix4x4[4]);  fprintf( stderr, "%f ; ",matrix4x4[5]);
  fprintf( stderr, "%f ",matrix4x4[6]);  fprintf( stderr, "%f ",matrix4x4[7]);  fprintf( stderr, "%f ]\n\n",matrix4x4[8]);

}


void copy3x3Matrix(double * out,double * in)
{
  out[0]=in[0];   out[1]=in[1];   out[2]=in[2];
  out[3]=in[3];   out[4]=in[4];   out[5]=in[5];
  out[6]=in[6];   out[7]=in[7];   out[8]=in[8];
}


void create3x3IdentityMatrix(double * m)
{
    m[0] = 1.0;  m[1] = 0.0;  m[2] = 0.0;
    m[3] = 0.0;  m[4] = 1.0;  m[5] = 0.0;
    m[6] = 0.0;  m[7] = 0.0;  m[8] = 1.0;
}


int upscale3x3to4x4(double * mat4x4,double * mat3x3)
{
  if  ( (mat3x3==0)||(mat4x4==0) )   { return 0; }

  //TRANSPOSED RESULT
  mat4x4[0]=mat3x3[0]; mat4x4[1]=mat3x3[1]; mat4x4[2]=mat3x3[2];  mat4x4[3]=0.0;
  mat4x4[4]=mat3x3[3]; mat4x4[5]=mat3x3[4]; mat4x4[6]=mat3x3[5];  mat4x4[7]=0.0;
  mat4x4[8]=mat3x3[6]; mat4x4[9]=mat3x3[7]; mat4x4[10]=mat3x3[8]; mat4x4[11]=0.0;
  mat4x4[12]=0.0;      mat4x4[13]=0.0;      mat4x4[14]=0.0;       mat4x4[15]=1.0;

  return 1;
}

int transpose3x3MatrixD(double * mat)
{
  if (mat==0) { return 0; }
  /*       -------  TRANSPOSE ------->
      0   1   2             0  3  6
      3   4   5             1  4  7
      6   7   8             2  5  8   */

  double tmp;
  tmp = mat[1]; mat[1]=mat[3];  mat[3]=tmp;
  tmp = mat[2]; mat[2]=mat[6];  mat[6]=tmp;
  tmp = mat[5]; mat[5]=mat[7];  mat[7]=tmp;
  return 1;
}


double det3x3Matrix(double * mat)
{
 double * a = mat;

 double detA  = a[I11] * a[I22] * a[I33];
        detA += a[I21] * a[I32] * a[I13];
        detA += a[I31] * a[I12] * a[I23];

 //FIRST PART DONE
        detA -= a[I11] * a[I32] * a[I23];
        detA -= a[I31] * a[I22] * a[I13];
        detA -= a[I21] * a[I12] * a[I33];

 return detA;
}


int invert3x3MatrixD(double * mat,double * result)
{
 double * a = mat;
 double * b = result;
 double detA = det3x3Matrix(mat);
 if (detA==0.0)
    {
      copy3x3Matrix(result,mat);
      fprintf(stderr,"Matrix 3x3 cannot be inverted (det = 0)\n");
      return 0;
    }
 double one_div_detA = (double) 1 / detA;

 //FIRST LINE
 b[I11]  = a[I22] * a[I33] - a[I23]*a[I32]  ;
 b[I11] *= one_div_detA;

 b[I12]  = a[I13] * a[I32] - a[I12]*a[I33]  ;
 b[I12] *= one_div_detA;

 b[I13]  = a[I12] * a[I23] - a[I13]*a[I22]  ;
 b[I13] *= one_div_detA;

 //SECOND LINE
 b[I21]  = a[I23] * a[I31] - a[I21]*a[I33]  ;
 b[I21] *= one_div_detA;

 b[I22]  = a[I11] * a[I33] - a[I13]*a[I31]  ;
 b[I22] *= one_div_detA;

 b[I23]  = a[I13] * a[I21] - a[I11]*a[I23]  ;
 b[I23] *= one_div_detA;

 //THIRD LINE
 b[I31]  = a[I21] * a[I32] - a[I22]*a[I31]  ;
 b[I31] *= one_div_detA;

 b[I32]  = a[I12] * a[I31] - a[I11]*a[I32]  ;
 b[I32] *= one_div_detA;

 b[I33]  = a[I11] * a[I22] - a[I12]*a[I21]  ;
 b[I33] *= one_div_detA;

 print3x3DMatrix("Inverted Matrix From Source",a);
 print3x3DMatrix("Inverted Matrix To Target",b);

 return 1;

}


int multiplyTwo3x3Matrices(double * result , double * matrixA , double * matrixB)
{
  if ( (matrixA==0) || (matrixB==0) || (result==0) ) { return 0; }

  fprintf(stderr,"Multiplying 3x3 A and B \n");
  print3x3DMatrix("A", matrixA);
  print3x3DMatrix("B", matrixB);

  //MULTIPLICATION_RESULT FIRST ROW
  // 0 1 2
  // 3 4 5
  // 6 7 8
  result[0]=matrixA[0] * matrixB[0] + matrixA[1] * matrixB[3]  + matrixA[2] * matrixB[6];
  result[1]=matrixA[0] * matrixB[1] + matrixA[1] * matrixB[4]  + matrixA[2] * matrixB[7];
  result[2]=matrixA[0] * matrixB[2] + matrixA[1] * matrixB[5]  + matrixA[2] * matrixB[8];

  result[3]=matrixA[3] * matrixB[0] + matrixA[4] * matrixB[3]  + matrixA[5] * matrixB[6];
  result[4]=matrixA[3] * matrixB[1] + matrixA[4] * matrixB[4]  + matrixA[5] * matrixB[7];
  result[5]=matrixA[3] * matrixB[2] + matrixA[4] * matrixB[5]  + matrixA[5] * matrixB[8];

  result[6]=matrixA[6] * matrixB[0] + matrixA[7] * matrixB[3]  + matrixA[8] * matrixB[6];
  result[7]=matrixA[6] * matrixB[1] + matrixA[7] * matrixB[4]  + matrixA[8] * matrixB[7];
  result[8]=matrixA[6] * matrixB[2] + matrixA[7] * matrixB[5]  + matrixA[8] * matrixB[8];

  print3x3DMatrix("AxB", result);

  return 1;
}
