/** @file matrixCalculations.h
 *  @brief  Functions to prepare matrixes and transform 3D points
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MATRIXCALCULATIONS_H_INCLUDED
#define MATRIXCALCULATIONS_H_INCLUDED



#ifdef __cplusplus
extern "C"
{
#endif



int convertRodriguezAndTranslationTo4x4DUnprojectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit);
int convertRodriguezAndTranslationToOpenGL4x4DProjectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit);

int convertTranslationTo4x4(double * translation, double * result);

int projectPointsFrom3Dto2D(double * x2D, double * y2D , double * x3D, double *y3D , double * z3D , double * intrinsics , double * rotation3x3 , double * translation);

void print4x4DMatrix(char * str , double * matrix4x4);



/**
* @brief build OpenGL Projection Matrix simulating a "real" camera
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting relative position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array 3x3 of object Rotation
* @param  Input Array 4x1 of absolute 3D position of the point ( X,Y,Z,W )
* @retval 0=Failure,1=Success
*/
void buildOpenGLProjectionForIntrinsics   (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             double nearPlane,
                                             double farPlane
                                           );

/**
* @brief Convert 3D Point in Relation to a 3D Object
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting relative position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array 3x3 of object Rotation
* @param  Input Array 4x1 of absolute 3D position of the point ( X,Y,Z,W )
* @retval 0=Failure,1=Success
*/
int pointInRelationToObject(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation3x3 , double * absoluteInPoint3DRotated );


int pointFromRelationToObjectXYZQuaternionXYZWToAbsolute(unsigned int method,  double * absoluteInPoint3DRotated , double * objectPosition , double * objectQuaternion ,double * relativeOutPoint3DUnrotated);

int normalizeQuaternions(double *qX,double *qY,double *qZ,double *qW);

void quaternion2Matrix3x3(double * matrix3x3,double * quaternions,int quaternionConvention);


int projectPointsFrom3Dto2D(double * x2D, double * y2D , double * x3D, double *y3D , double * z3D , double * intrinsics , double * rotation3x3 , double * translation);

int convertRodriguezAndTranslationTo4x4DUnprojectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit);


int convertRodriguezAndTranslationToOpenGL4x4DProjectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit );


int move3DPoint(double * resultPoint3D, double * transformation4x4, double * point3D  );


void buildOpenGLProjectionForIntrinsics   (
                                             double * frustum,
                                             int * viewport ,
                                             double fx,
                                             double fy,
                                             double skew,
                                             double cx, double cy,
                                             unsigned int imageWidth, unsigned int imageHeight,
                                             double nearPlane,
                                             double farPlane
                                           );



int pointFromRelationWithObjectToAbsolute(double * absoluteOutPoint3DRotated, double * objectPosition , double * objectRotation3x3 ,  double * relativeInPoint3DUnrotated);

int pointFromAbsoluteToInRelationWithObject(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation3x3 , double * absoluteInPoint3DRotated );
int pointFromAbsoluteToInRelationWithObject_UsingInversion(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation3x3 , double * absoluteInPoint3DRotated );

int pointFromAbsoluteToRelationWithObject_PosXYZRotationXYZ(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation , double * absoluteInPoint3DRotated );
int pointFromAbsoluteToRelationWithObject_PosXYZQuaternionXYZW(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectQuaternion , double * absoluteInPoint3DRotated );

int pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(double * absoluteOutPoint3DRotated , double * objectPosition , double * objectRotation ,double * relativeInPoint3DUnrotated);
int pointFromRelationWithObjectToAbsolute_PosXYZQuaternionXYZW(double * absoluteOutPoint3DRotated , double * objectPosition , double * objectQuaternion ,double * relativeInPoint3DUnrotated);





/*
  TAKEN FROM http://www.lighthouse3d.com/opengl/maths/index.php?raytriint

*/


#define innerProduct(v,q) \
       ((v)[0] * (q)[0] + \
		(v)[1] * (q)[1] + \
		(v)[2] * (q)[2])


#define crossProduct(a,b,c) \
        (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2]; \
        (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0]; \
        (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];



/* a = b - c */
#define vector(a,b,c) \
        (a)[0] = (b)[0] - (c)[0];	\
        (a)[1] = (b)[1] - (c)[1];	\
        (a)[2] = (b)[2] - (c)[2];


int rayIntersectsTriangle(float *p, float *d,float *v0, float *v1, float *v2);

int rayIntersectsRectangle(float *p, float *d,float *v0, float *v1, float *v2, float *v3);
//http://ilab.usc.edu/wiki/index.php/Fast_Square_Root
inline float sqrt_fast_approximation(const float x);
double distanceBetween3DPoints(double * p1, double * p2);
float distanceBetween3DPointsFast(float *x1,float*y1,float *z1,float *x2,float*y2,float *z2);
float squaredDistanceBetween3DPoints(float *x1,float*y1,float *z1,float *x2,float*y2,float *z2);

/**
* @brief Calculate the distance between two 3d points
* @ingroup OGLTools
* @param Point 1 - X
* @param Point 1 - Y
* @param Point 1 - Z
* @param Point 2 - X
* @param Point 2 - Y
* @param Point 2 - Z
* @retval A float describing the distance
*/
float calculateDistance(float from_x,float from_y,float from_z,float to_x,float to_y,float to_z);
void vectorDirection(float src_x,float src_y,float src_z,float targ_x,float targ_y,float targ_z,float *vect_x,float *vect_y,float *vect_z);
void findNormal(float *v1x, float *v1y, float *v1z, float v2x, float v2y, float v2z, float v3x, float v3y, float v3z );


/**
* @brief Internally Test Matrix subsystem
* @ingroup AmMatrix
*/
void testMatrices();



#ifdef __cplusplus
}
#endif


#endif // MATRIXCALCULATIONS_H_INCLUDED
