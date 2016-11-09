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
* @author Greg James - gjames@NVIDIA.com , Lisc: Free code - no warranty & no money back.  Use it all you want
* @ingroup AmMatrix
* @brief
*    This function tests if the 3D point 'testpt' lies within an arbitrarily
* oriented cylinder.  The cylinder is defined by an axis from 'pt1' to 'pt2',
* the axis having a length squared of 'lengthsq' (pre-compute for each cylinder
* to avoid repeated work!), and radius squared of 'radius_sq'.
*    The function tests against the end caps first, which is cheap -> only
* a single dot product to test against the parallel cylinder caps.  If the
* point is within these, more work is done to find the distance of the point
* from the cylinder axis.
*    Fancy Math (TM) makes the whole test possible with only two dot-products
* a subtract, and two multiplies.  For clarity, the 2nd mult is kept as a
* divide.  It might be faster to change this to a mult by also passing in
* 1/lengthsq and using that instead.
*    Elminiate the first 3 subtracts by specifying the cylinder as a base
* point on one end cap and a vector to the other end cap (pass in {dx,dy,dz}
* instead of 'pt2' ).
*
*    The dot product is constant along a plane perpendicular to a vector.
*    The magnitude of the cross product divided by one vector length is
* constant along a cylinder surface defined by the other vector as axis.
*
* @param  Cylinder axis starting Point ( xyz )
* @param  Cylinder axis ending Point ( xyz )
* @param  The length of the cylinder squared
* @param  The radius of the cylinder squared
* @param  The point to test if it is inside or out
* @retval  -1.0 if point is outside the cylinder or the distance squared from cylinder axis if point is inside.
*/
float cylinderTest( float * pt1, float * pt2, float lengthsq, float radius_sq, float * testpt );

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


/**
* @brief Convert 3D Point From in Relation to a 3D Object , to an Absolute Coordinate System using a 3x3 Matrix
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting absolute position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array 3x3 of a Rotation Matrix
* @param  Input Array 4x1 of relative 3D position of the point we want to convert ( X,Y,Z,W )
* @retval 0=Failure,1=Success */
int pointFromRelationWithObjectToAbsolute(double * absoluteOutPoint3DRotated, double * objectPosition , double * objectRotation3x3 ,  double * relativeInPoint3DUnrotated);

/**
* @brief Convert 3D Point From an Absolute Coordinate System , to in Relation with a 3D Object using a 3x3 Matrix
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting relative position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array 3x3 of a Rotation Matrix
* @param  Input Array 4x1 of absolute 3D position of the point we want to convert ( X,Y,Z,W )
* @retval 0=Failure,1=Success */
int pointFromAbsoluteToInRelationWithObject(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation3x3 , double * absoluteInPoint3DRotated );



/**
* @brief Convert 3D Point From an Absolute Coordinate System , to in Relation with a 3D Object using an Euler Rotation
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting relative position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array of Euler Angles
* @param  Input Array 4x1 of absolute 3D position of the point we want to convert ( X,Y,Z,W )
* @retval 0=Failure,1=Success */
int pointFromAbsoluteToRelationWithObject_PosXYZRotationXYZ(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation , double * absoluteInPoint3DRotated );

/**
* @brief Convert 3D Point From an Absolute Coordinate System , to in Relation with a 3D Object using a Quaternion Rotation
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting relative position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array Quaternion qX qY qZ qW order
* @param  Input Array 4x1 of absolute 3D position of the point we want to convert ( X,Y,Z,W )
* @retval 0=Failure,1=Success */
int pointFromAbsoluteToRelationWithObject_PosXYZQuaternionXYZW(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectQuaternion , double * absoluteInPoint3DRotated );


/**
* @brief Convert 3D Point From in Relation to a 3D Object , to an Absolute Coordinate System using an Euler Rotation
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting absolute position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array of Euler Angles
* @param  Input Array 4x1 of relative 3D position of the point we want to convert ( X,Y,Z,W )
* @retval 0=Failure,1=Success */
int pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(double * absoluteOutPoint3DRotated , double * objectPosition , double * objectRotation ,double * relativeInPoint3DUnrotated);



/**
* @brief Convert 3D Point From in Relation to a 3D Object , to an Absolute Coordinate System using a Quaternion Rotation
* @ingroup AmMatrix
* @param  Output Array 4x1 of resulting absolute position ( X,Y,Z,W )
* @param  Input Array 4x1 of object Position ( X,Y,Z,W )
* @param  Input Array Quaternion qX qY qZ qW order
* @param  Input Array 4x1 of relative 3D position of the point we want to convert ( X,Y,Z,W )
* @retval 0=Failure,1=Success */
int pointFromRelationWithObjectToAbsolute_PosXYZQuaternionXYZW(double * absoluteOutPoint3DRotated , double * objectPosition , double * objectQuaternion ,double * relativeInPoint3DUnrotated);






/**
 * @brief Perform Slerp function between 2 4x4 matrices ( of doubles ) representing rigid transformations
 * @ingroup AmMatrix
 * @param Output Matrix4x4  ( of doubles )
 * @param Input Matrix4x4 A ( of doubles )
 * @param Input Matrix4x4 B ( of doubles )
 * @param Factor , typically should be 0.5 for half and half
 */
int slerp2RotTransMatrices4x4(double * result4, double * a4, double * b4 , float step );




/**
 * @brief Perform Slerp function between 2 4x4 matrices ( of floats ) representing rigid transformations
 * @ingroup AmMatrix
 * @param Output Matrix4x4   ( of floats )
 * @param Input Matrix4x4 A  ( of floats )
 * @param Input Matrix4x4 B  ( of floats )
 * @param Factor , typically should be 0.5 for half and half
 */
int slerp2RotTransMatrices4x4F(float * result4, float * a4, float * b4 , float step );

/**
* @brief Return the Inner Product of 2 3D points
* @ingroup AmMatrix
* @param  Input Array A (XYZ)
* @param  Input Array B (XYZ)
* @retval Result number of the inner product
*/
#define innerProduct(v,q) \
       ((v)[0] * (q)[0] + \
		(v)[1] * (q)[1] + \
		(v)[2] * (q)[2])

/**
* @brief Return the Cross Product of 2 3D points
* @ingroup AmMatrix
* @param  Output Array  (XYZ)
* @param  Input Point Array A (XYZ)
* @param  Input Point Array B (XYZ)
*/
#define crossProduct(a,b,c) \
        (a)[0] = (b)[1] * (c)[2] - (c)[1] * (b)[2]; \
        (a)[1] = (b)[2] * (c)[0] - (c)[2] * (b)[0]; \
        (a)[2] = (b)[0] * (c)[1] - (c)[0] * (b)[1];

/**
* @brief Convert 2 Points to a Vector ( a = b - c )
* @ingroup AmMatrix
* @param  Output Array  (XYZ)
* @param  Input Point Array A (XYZ)
* @param  Input Point Array B (XYZ)
*/
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
  TAKEN FROM   http://www.lighthouse3d.com/tutorials/maths/ray-triangle-intersection/
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
//static inline float sqrt_fast_approximation(const float x);
//http://ilab.usc.edu/wiki/index.php/Fast_Square_Root
static inline float sqrt_fast_approximation(const float x)
{
  union
  {
    int i;
    float x;
  } u;

  u.x = x;
  u.i = (1<<29) + (u.i >> 1) - (1<<22);
  return u.x;
}





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



/**
* @brief Convert 2 Points to a Vector Direction ( a = b - c )
* @ingroup AmMatrix
* @param  Input Point Array A X
* @param  Input Point Array A Y
* @param  Input Point Array A Z
* @param  Input Point Array B X
* @param  Input Point Array B Y
* @param  Input Point Array B Z
* @param  Output Array X
* @param  Output Array Y
* @param  Output Array Z
*/
void vectorDirection(float src_x,float src_y,float src_z,float targ_x,float targ_y,float targ_z,float *vect_x,float *vect_y,float *vect_z);



/**
* @brief Calculate the normal between three 3d points
* @ingroup OGLTools
* @param Input Point 1/Output X
* @param Input Point 1/Output Y
* @param Input Point 1/Output Z
* @param Point 2 - X
* @param Point 2 - Y
* @param Point 2 - Z
* @param Point 3 - X
* @param Point 3 - Y
* @param Point 3 - Z
* @retval A float describing the distance
*/
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
