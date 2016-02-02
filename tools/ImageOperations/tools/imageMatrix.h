#ifndef IMAGEMATRIX_H_INCLUDED
#define IMAGEMATRIX_H_INCLUDED

int castUCharImage2FloatAndNormalize(float * out , unsigned char * in, unsigned int width,unsigned int height , unsigned int channels);

int castUCharImage2Float(float * out , unsigned char * in, unsigned int width,unsigned int height , unsigned int channels);
int castFloatImage2UChar(unsigned char * out , float * in, unsigned int width,unsigned int height , unsigned int channels);
float * copyUCharImage2Float(unsigned char * in, unsigned int width,unsigned int height , unsigned int channels);

unsigned char* divideTwoImages(unsigned char *  dividend , unsigned char * divisor , unsigned int width,unsigned int height , unsigned int channels);
int divide2DMatricesF(float * out , float * dividend , float * divisor , unsigned int width , unsigned int height , unsigned int channels);
int multiply2DMatricesF(float * out , float * mult1 , float * mult2 , unsigned int width , unsigned int height, unsigned int channels );
int multiply2DMatricesFWithUC(float * out , float * mult1 , unsigned char * mult2 , unsigned int width , unsigned int height, unsigned int channels );


#endif // IMAGEMATRIX_H_INCLUDED
