#ifndef MATRIX4X4TOOLS_H_INCLUDED
#define MATRIX4X4TOOLS_H_INCLUDED

double * alloc4x4Matrix();
void free4x4Matrix(double ** mat);

void print4x4FMatrix(char * str , float * matrix4x4);
void print4x4DMatrix(char * str , double * matrix4x4);
void copy4x4Matrix(double * out,double * in) ;
void create4x4IdentityMatrix(double * m) ;
void create4x4RotationMatrix(double * m,double angle, double x, double y, double z) ;
void create4x4TranslationMatrix(double * matrix,double x, double y, double z);
void create4x4ScalingMatrix(double * matrix,double sx, double sy, double sz);
void create4x4RotationX(double * matrix,double degrees) ;
void create4x4RotationY(double * matrix,double degrees) ;
void create4x4RotationZ(double * matrix,double degrees) ;
double det4x4Matrix(double * mat) ;
int invert4x4MatrixD(double * result,double * mat) ;
int transpose4x4MatrixD(double * mat) ;
int multiplyTwo4x4Matrices(double * result , double * matrixA , double * matrixB);

#endif // MATRIX4X4TOOLS_H_INCLUDED
