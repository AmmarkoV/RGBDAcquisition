#ifndef MATRIXCALCULATIONS_H_INCLUDED
#define MATRIXCALCULATIONS_H_INCLUDED


int convertRodriguezAndTransTo4x4(float * rodriguez , float * translation , float * matrix4x4 );
int convertTranslationTo4x4(float * translation, float * result);

void print4x4DMatrix(char * str , double * matrix4x4);

void buildOpenGLProjectionForIntrinsics   (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             int width, int height,
                                             double nearPlane,
                                             double farPlane
                                           );

void testMatrices();

#endif // MATRIXCALCULATIONS_H_INCLUDED
