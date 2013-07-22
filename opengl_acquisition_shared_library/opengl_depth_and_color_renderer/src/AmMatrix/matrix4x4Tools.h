#ifndef MATRIX4X4TOOLS_H_INCLUDED
#define MATRIX4X4TOOLS_H_INCLUDED

void print4x4DMatrix(char * str , double * matrix4x4);
void copy4x4Matrix(double * out,double * in) ;
void create4x4IdentityMatrix(double * m) ;
void create4x4RotationMatrix(double angle, double x, double y, double z ,  double * m) ;
void create4x4TranslationMatrix(double x, double y, double z, double * matrix)  ;
void create4x4ScalingMatrix(double sx, double sy, double sz, double * matrix) ;
void create4x4RotationX(double degrees, double * matrix) ;
void create4x4RotationY(double degrees, double * matrix) ;
void create4x4RotationZ(double degrees, double * matrix) ;
int det4x4Matrix(double * mat) ;
int invert4x4MatrixD(double * mat,double * result) ;
int transpose4x4MatrixD(double * mat) ;
int multiplyTwo4x4Matrices(double * result , double * matrixA , double * matrixB);

#endif // MATRIX4X4TOOLS_H_INCLUDED
