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

/**
* @brief build Projection Matrix using Rodriguez Rotation and a translation ( typically coming from OpenCV )
* @ingroup AmMatrix
* @param  Output Array 4x4 of resulting matrix
* @param  Input Rodriguez Coordinates
* @param  Input Translation Coordinate
* @param  Input Unit Scale
* @retval 0=Failure,1=Success
*/
int convertRodriguezAndTranslationTo4x4DUnprojectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit);

/**
* @brief build OpenGL Projection Matrix using Rodriguez Rotation and a translation ( typically coming from OpenCV )
* @ingroup AmMatrix
* @param  Output Array 4x4 of resulting matrix
* @param  Input Rodriguez Coordinates
* @param  Input Translation Coordinate
* @param  Input Unit Scale
* @retval 0=Failure,1=Success
*/
int convertRodriguezAndTranslationToOpenGL4x4DProjectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit);


/**
* @brief Project a 3D Point to a 2D surface , emulating a camera
* @ingroup AmMatrix
* @param  Output 2D X Point
* @param  Output 2D Y Point

* @param  Input 3D X Point
* @param  Input 3D Y Point
* @param  Input 3D Z Point


* @param  Input Intrinsics Matrix

* @param  Input Rotation 3x3 Matrix
* @param  Input Translation 3x1
* @retval 0=Failure,1=Success
*/
int projectPointsFrom3Dto2D(double * x2D, double * y2D , double * x3D, double *y3D , double * z3D , double * intrinsics , double * rotation3x3 , double * translation);




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
int pointFromRelationToObjectXYZQuaternionXYZWToAbsolute(unsigned int method,  double * absoluteInPoint3DRotated , double * objectPosition , double * objectQuaternion ,double * relativeOutPoint3DUnrotated);



/**
* @brief Multiply a 4x4 matrix with a Vector (3D Point)  A*V
* @ingroup AmMatrix
* @param  Output Vector ( should be already allocated )
* @param  Input 4x4 Matrix A
* @param  Input Vector 4x1 V
* @retval 0=failure,1=success
*/
int move3DPoint(double * resultPoint3D, double * transformation4x4, double * point3D  );





int pointFromRelationWithObjectToAbsolute(double * absoluteOutPoint3DRotated, double * objectPosition , double * objectRotation3x3 ,  double * relativeInPoint3DUnrotated);
int pointFromAbsoluteToInRelationWithObject(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation3x3 , double * absoluteInPoint3DRotated );

int pointFromAbsoluteToRelationWithObject_PosXYZRotationXYZ(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation , double * absoluteInPoint3DRotated );
int pointFromAbsoluteToRelationWithObject_PosXYZQuaternionXYZW(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectQuaternion , double * absoluteInPoint3DRotated );

int pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(double * absoluteOutPoint3DRotated , double * objectPosition , double * objectRotation ,double * relativeInPoint3DUnrotated);
int pointFromRelationWithObjectToAbsolute_PosXYZQuaternionXYZW(double * absoluteOutPoint3DRotated , double * objectPosition , double * objectQuaternion ,double * relativeInPoint3DUnrotated);





/*
  TAKEN FROM http://www.lighthouse3d.com/opengl/maths/index.php?raytriint
  http://www.lighthouse3d.com/tutorials/maths/ray-triangle-intersection/
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


/**
* @brief Check if a Ray Intersects a Triangle
* @ingroup AmMatrix
* @param  Beginning Point for Ray
* @param  Direction of Ray ?
* @param  3D Position of Triangle point V0
* @param  3D Position of Triangle point V1
* @param  3D Position of Triangle point V2
* @retval 0=NoIntersection,1=Intersects
*/
int rayIntersectsTriangle(float *p, float *d,float *v0, float *v1, float *v2);

/**
* @brief Check if a Ray Intersects a Rectangle
* @ingroup AmMatrix
* @param  Beginning Point for Ray
* @param  Direction of Ray ?
* @param  3D Position of Triangle point V0
* @param  3D Position of Triangle point V1
* @param  3D Position of Triangle point V2
* @param  3D Position of Triangle point V3
* @retval 0=NoIntersection,1=Intersects
*/
int rayIntersectsRectangle(float *p, float *d,float *v0, float *v1, float *v2, float *v3);



/**
* @brief Check if a Ray Intersects a Rectangle
* @ingroup AmMatrix
* @param  Number which we are searching for its square root
* @retval Square Root Using a fast approximation
* @bug Using an approximation of the sqrt is fast , but it should only be used for approximations and not accurate results
Found at : http://ilab.usc.edu/wiki/index.php/Fast_Square_Root
*/
inline float sqrt_fast_approximation(const float x);


/**
* @brief Count the distance between 2 points
* @ingroup AmMatrix
* @param  Point A
* @param  Point B
* @retval Distance between the two points
*/
double distanceBetween3DPoints(double * p1, double * p2);

/**
* @brief Find an approximation of the distance between 2 points
* @ingroup AmMatrix
* @param  Point A
* @param  Point B
* @retval Distance between the two points
* @bug Using an approximation of the distance between 2 points is fast , but it should only be used for approximations and not accurate results
*/
float distanceBetween3DPointsFast(float *x1,float*y1,float *z1,float *x2,float*y2,float *z2);


/**
* @brief Count the squared distance between 2 points
* @ingroup AmMatrix
* @param  Point A
* @param  Point B
* @retval Squared Distance between the two points
*/
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
