#ifndef IMAGEMATRIX_H_INCLUDED
#define IMAGEMATRIX_H_INCLUDED


unsigned char* divideTwoImages1Ch(unsigned char *  divisor , unsigned char * divider , unsigned int width,unsigned int height );
int divide2DMatricesF(float * out , float * divider , float * divisor , unsigned int width , unsigned int height );
int multiply2DMatricesF(float * out , float * mult1 , float * mult2 , unsigned int width , unsigned int height );
int multiply2DMatricesFWithUC(float * out , float * mult1 , unsigned char * mult2 , unsigned int width , unsigned int height );


#endif // IMAGEMATRIX_H_INCLUDED
