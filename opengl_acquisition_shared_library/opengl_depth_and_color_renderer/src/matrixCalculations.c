#include "matrixCalculations.h"
#include <stdio.h>

int upscale3x3to4x4(float * mat3x3,float * mat4x4)
{
  if  ( (mat3x3==0)||(mat4x4==0) )   { return 0; }

  //TRANSPOSED RESULT
  mat4x4[0]=mat3x3[0]; mat4x4[1]=mat3x3[1]; mat4x4[2]=mat3x3[2];  mat4x4[3]=0.0;
  mat4x4[4]=mat3x3[3]; mat4x4[5]=mat3x3[4]; mat4x4[6]=mat3x3[5];  mat4x4[7]=0.0;
  mat4x4[8]=mat3x3[6]; mat4x4[9]=mat3x3[7]; mat4x4[10]=mat3x3[8]; mat4x4[11]=0.0;
  mat4x4[12]=0.0;      mat4x4[13]=0.0;      mat4x4[14]=0.0;       mat4x4[15]=1.0;

  return 1;
}


int convertRodriguezTo3x3(float * rodriguez , float * result)
{
  if ( (rodriguez==0) ||  (result==0) ) { return 0; }
  float x = rodriguez[0] , y = rodriguez[1] , z = rodriguez[2];
  float th = sqrt( x*x + y*y + z*z );
  float cosTh = cos(th);
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

int convertTranslationTo4x4(float * translation, float * result)
{
  if ( (translation==0) ||  (result==0) ) { return 0; }
  float x = translation[0] , y = translation[1] , z = translation[2];

  result[0]=1.0; result[1]=0;   result[2]=0;    result[3]=x;
  result[4]=0;   result[5]=1.0; result[6]=0;    result[7]=y;
  result[8]=0;   result[9]=0;   result[10]=1.0; result[11]=z;
  result[12]=0;  result[13]=0;  result[14]=0;   result[15]=1.0;

  return 1;
}



int multiplyVectorWith3x3Matrix(float * matrix, float * result)
{
  if ( (matrix==0) ||  (result==0) ) { return 0; }
  float x = matrix[0] , y = matrix[1] , z = matrix[2];
  float th = sqrt( x*x + y*y + z*z );
  float cosTh = cos(th);
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

int convertRodriguezAndTransTo4x4(float * rodriguez , float * translation , float * matrix4x4 )
{
  //return 0;
  float matrix3x3[9]={0};
  convertRodriguezTo3x3(rodriguez,(float*) matrix3x3);
  upscale3x3to4x4((float*) matrix3x3,matrix4x4);
  matrix4x4[3]=translation[0]; matrix4x4[7]=translation[1]; matrix4x4[11]=translation[2];

  //convertTranslationTo4x4(translation,matrix4x4);

  fprintf( stderr, "  %f ",matrix4x4[0]);  fprintf( stderr, "%f ",matrix4x4[1]);  fprintf( stderr, "%f ",matrix4x4[2]);  fprintf( stderr, "%f\n",matrix4x4[3]);
  fprintf( stderr, "  %f ",matrix4x4[4]);  fprintf( stderr, "%f ",matrix4x4[5]);  fprintf( stderr, "%f ",matrix4x4[6]);  fprintf( stderr, "%f\n",matrix4x4[7]);
  fprintf( stderr, "  %f ",matrix4x4[8]);  fprintf( stderr, "%f ",matrix4x4[9]);  fprintf( stderr, "%f ",matrix4x4[10]); fprintf( stderr, "%f\n",matrix4x4[11]);
  fprintf( stderr, "  %f ",matrix4x4[12]); fprintf( stderr, "%f ",matrix4x4[13]); fprintf( stderr, "%f ",matrix4x4[14]); fprintf( stderr, "%f\n",matrix4x4[15]);

 return 1;
}

