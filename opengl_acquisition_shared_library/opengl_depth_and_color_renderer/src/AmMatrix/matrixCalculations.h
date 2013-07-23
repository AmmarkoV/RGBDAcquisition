#ifndef MATRIXCALCULATIONS_H_INCLUDED
#define MATRIXCALCULATIONS_H_INCLUDED


int convertRodriguezAndTransTo4x4(double * result4x4, double * rodriguez , double * translation );
int convertTranslationTo4x4(double * translation, double * result);

void print4x4DMatrix(char * str , double * matrix4x4);

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
                                           );

void testMatrices();

#endif // MATRIXCALCULATIONS_H_INCLUDED
