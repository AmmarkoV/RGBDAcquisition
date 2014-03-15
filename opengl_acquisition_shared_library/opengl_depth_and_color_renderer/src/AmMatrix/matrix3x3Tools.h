/** @file matrix3x3Tools.h
 *  @brief  A small 3x3 matrix library for simple 3x3 transformations
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MATRIX3X3TOOLS_H_INCLUDED
#define MATRIX3X3TOOLS_H_INCLUDED

double * alloc3x3Matrix();
void free3x3Matrix(double ** mat);

void print3x3FMatrix(char * str , float * matrix3x3);
void print3x3DMatrix(char * str , double * matrix3x3);
void print3x3DScilabMatrix(char * str , double * matrix3x3);

void copy3x3Matrix(double * out,double * in);
void create3x3IdentityMatrix(double * m);

int transpose3x3MatrixD(double * mat);

int upscale3x3to4x4(double * mat4x4,double * mat3x3);


int random3x3Matrix(double * mat,double minimumValues, double maximumValues);

double det3x3Matrix(double * mat);
int invert3x3MatrixD(double * mat,double * result);

int multiplyTwo3x3Matrices(double * result , double * matrixA , double * matrixB);

int transform2DPointUsing3x3Matrix(double * resultPoint2D, double * transformation3x3, double * point2D);


#endif // MATRIX3X3TOOLS_H_INCLUDED
