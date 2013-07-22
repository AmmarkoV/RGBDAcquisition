#include "matrix3x3Tools.h"
#include <stdio.h>
#include <math.h>

void print3x3DMatrix(char * str , double * matrix4x4)
{
  fprintf( stderr, "  3x3 double %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf( stderr, "%f ",matrix4x4[0]);  fprintf( stderr, "%f ",matrix4x4[1]);  fprintf( stderr, "%f\n",matrix4x4[2]);
  fprintf( stderr, "%f ",matrix4x4[3]);  fprintf( stderr, "%f ",matrix4x4[4]);  fprintf( stderr, "%f\n",matrix4x4[5]);
  fprintf( stderr, "%f ",matrix4x4[6]);  fprintf( stderr, "%f ",matrix4x4[7]);  fprintf( stderr, "%f\n",matrix4x4[8]);
  fprintf( stderr, "--------------------------------------\n");
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
