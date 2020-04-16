/** @file matrix4x4Tools.h
 *  @brief  A small 4x4 matrix library for simple 4x4 transformations
 *  @author Ammar Qammaz (AmmarkoV)
 */

#ifndef MATRIX4X4TOOLS_H_INCLUDED
#define MATRIX4X4TOOLS_H_INCLUDED


#ifdef __cplusplus
extern "C"
{
#endif

/**
* @brief Allocate a new 4x4 Matrix
* @ingroup AmMatrix
* @retval 0=Failure or a pointer to an allocated 3x3 Matrix
*/
double * malloc4x4DMatrix();

/**
* @brief Deallocate an existing 3x3 Matrix
* @ingroup AmMatrix
* @param  Pointer to a Pointer of an allocated matrix
*/
void free4x4DMatrix(double ** mat);



/**
* @brief Printout an 4x4 Matrix that consists of floats
* @ingroup AmMatrix
* @param  Label for the printout ( cString )
* @param  Pointer to a Pointer of an allocated float matrix
*/
void print4x4FMatrix(const char * str , float * matrix4x4,int forcePrint);

/**
* @brief Printout an 4x4 Matrix that consists of doubles
* @ingroup AmMatrix
* @param  Label for the printout ( cString )
* @param  Pointer to a Pointer of an allocated doubles matrix
*/
void print4x4DMatrix(const char * str , double * matrix4x4,int forcePrint);



/**
* @brief Copy a 4x4 Matrix of doubles to another
* @ingroup AmMatrix
* @param  Output Matrix
* @param  Input Matrix
* @param  Pointer to a Pointer of an allocated doubles matrix
*/
void copy4x4DMatrix(double * out,double * in);




void copy3x3FMatrixTo4x4F(float * out,float * in);

/**
* @brief Copy a 4x4 Matrix of floats to another
* @ingroup AmMatrix
* @param  Output Matrix
* @param  Input Matrix
* @param  Pointer to a Pointer of an allocated doubles matrix
*/
void copy4x4FMatrix(float * out,float * in);

/**
* @brief Convert a 4x4 Matrix from Float To Double
* @ingroup AmMatrix
* @param  Pointer to a double 4x4 output
* @param  Pointer to a float 4x4 input
*/
void copy4x4FMatrixTo4x4D(double * out,float * in);


/**
* @brief Convert a 4x4 Matrix from Double To Float
* @ingroup AmMatrix
* @param  Pointer to a float 4x4 output
* @param  Pointer to a double 4x4 input
*/
void copy4x4DMatrixTo4x4F(float * dest, double * m );

/**
* @brief Set an allocated 4x4 matrix to Identity ( diagonal 1 , all else 0 )
* @ingroup AmMatrix
* @param  Input/Output Matrix
*/
void create4x4DIdentityMatrix(double * m) ;

void create4x4FIdentityMatrix(float * m);

int is4x4DZeroMatrix(double  * m);

int is4x4DIdentityMatrix(double * m);


int is4x4FIdentityMatrix(float  * m);
int is4x4FIdentityMatrixPercisionCompensating(float  * m);


void convert4x4DMatrixToRPY(double *m ,double *roll,double *pitch,double *yaw);




/**
* @brief Convert an allocated 4x4 matrix to a homogeneous Translation rotation
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Input angle
* @param  X Axis Parameter
* @param  Y Axis Parameter
* @param  Z Axis Parameter
*/
void create4x4DRotationMatrix(double * m,double angle, double x, double y, double z) ;


void create4x4FRotationMatrix(float * m , float angle, float x, float y, float z);


void create4x4FTranslationMatrix(float * matrix , float x, float y, float z);


/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Translation
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  X Translation
* @param  Y Translation
* @param  Z Translation
*/
void create4x4DTranslationMatrix(double * matrix,double x, double y, double z);


/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Scaling
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Scaling on the X
* @param  Scaling on the Y
* @param  Scaling on the Z
* @retval 0=Failure,1=Success
*/
void create4x4DScalingMatrix(double * matrix,double scaleX, double scaleY, double scaleZ);

void create4x4FScalingMatrix(float * matrix , float scaleX, float scaleY, float scaleZ);


/**
* @brief Convert a quaternion to 4x4 matrix
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Quaternion on the X
* @param  Quaternion on the Y
* @param  Quaternion on the Z
* @param  Quaternion on the W
* @param  Degrees of rotation
*/
void create4x4DQuaternionMatrix(double * m , double qX,double qY,double qZ,double qW);




//This should be ROTATION_ORDER_NAMESA but it isn't to avoid bugs
//Since this used to be a variable in some points of the code..
static const char * ROTATION_ORDER_NAMESA[] =
{
  "ROTATION_ORDER_NONE", //0
  "ROTATION_ORDER_XYZ",//1
  "ROTATION_ORDER_XZY",//2
  "ROTATION_ORDER_YXZ",//3
  "ROTATION_ORDER_YZX",//4
  "ROTATION_ORDER_ZXY",//5
  "ROTATION_ORDER_ZYX",//6
  "ROTATION_ORDER_RPY",//7
  //--------------------
  "INVALID_ROTATION_ORDER"
};



enum ROTATION_ORDER
{
  ROTATION_ORDER_NONE=0,
  ROTATION_ORDER_XYZ,//1
  ROTATION_ORDER_XZY,//2
  ROTATION_ORDER_YXZ,//3
  ROTATION_ORDER_YZX,//4
  ROTATION_ORDER_ZXY,//5
  ROTATION_ORDER_ZYX,//6
  ROTATION_ORDER_RPY,//7
  //--------------------
  ROTATION_ORDER_NUMBER_OF_NAMES
};


/**
* @brief Convert euler angles in degrees to a 4x4 rotation matrix using any rotation order wanted
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Rotation X in euler angles (0-360)
* @param  Rotation Y in euler angles (0-360)
* @param  Rotation Z in euler angles (0-360)
* @param  Rotation order given by enum ROTATION_ORDER, typically ROTATION_ORDER_ZYX or ROTATION_ORDER_XYZ
*/
void create4x4DMatrixFromEulerAnglesWithRotationOrder(double * m ,double eulX, double eulY, double eulZ,unsigned int rotationOrder);


void create4x4FMatrixFromEulerAnglesWithRotationOrder(float * m ,float eulX, float eulY, float eulZ,unsigned int rotationOrder);


/**
* @brief Convert a quaternion to 4x4 matrix ZYX convention ( standard )
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Rotation X in euler angles (0-360)
* @param  Rotation Y in euler angles (0-360)
* @param  Rotation Z in euler angles (0-360)
*/
void create4x4DMatrixFromEulerAnglesZYX(double * m ,double eulX, double eulY, double eulZ);


/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Rotation on the X axis
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Degrees of rotation
*/
void create4x4DRotationX(double * matrix,double degrees) ;

/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Rotation on the Y axis
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Degrees of rotation
*/
void create4x4DRotationY(double * matrix,double degrees) ;

/**
* @brief Convert an allocated 4x4 matrix to a homogeneous 3D Rotation on the Z axis
* @ingroup AmMatrix
* @param  Output already allocated 4x4 Matrix
* @param  Degrees of rotation
*/
void create4x4DRotationZ(double * matrix,double degrees);


/**
* @brief Compute the determenant of a 4x4 matrix
* @ingroup AmMatrix
* @param  Input 4x4 Matrix
* @retval Det(mat)
*/
double det4x4DMatrix(double * mat) ;

/**
* @brief Invert a 4x4 matrix
* @ingroup AmMatrix
* @param  Input 4x4 Matrix
* @param  Output ( should be already allocated ) 3x3 Matrix
* @retval 0=failure,1=success
*/
int invert4x4DMatrix(double * result,double * mat) ;


int transpose4x4FMatrix(float * mat);

/**
* @brief Transpose an allocated 4x4 matrix to Identity ( diagonal 1 , all else 0 )
* @ingroup AmMatrix
* @param  Input/Output Matrix
* @retval 0=Failure,1=Success
*/
int transpose4x4DMatrix(double * mat) ;


/**
* @brief Multiply 2x 4x4 matrices ( A * B )
* @ingroup AmMatrix
* @param  Output 4x4 Matrix ( should be already allocated )
* @param  Input 4x4 Matrix A
* @param  Input 4x4 Matrix B
* @retval 0=failure,1=success
*/
int multiplyTwo4x4DMatrices(double * result , double * matrixA , double * matrixB);


int multiplyTwo4x4DMatricesBuffered(double * result , double * matrixA , double * matrixB);

int multiplyThree4x4DMatrices(double * result , double * matrixA , double * matrixB , double * matrixC);

int multiplyFour4x4DMatrices(double * result , double * matrixA , double * matrixB , double * matrixC , double * matrixD);

/**
* @brief Multiply 2x 4x4 Float matrices ( A * B )
* @ingroup AmMatrix
* @param  Output 4x4 Float Matrix ( should be already allocated )
* @param  Input 4x4 Float Matrix A
* @param  Input 4x4 Float Matrix B
* @retval 0=failure,1=success
*/
int multiplyTwo4x4FMatrices(float * result , float * matrixA , float * matrixB);


int multiplyTwo4x4FMatricesBuffered(float * result , float * matrixA , float * matrixB);

int multiplyThree4x4FMatrices(float * result , float * matrixA , float * matrixB , float * matrixC);

/**
* @brief Multiply a 4x4 matrix of doubles with a double precision Vector (3D Point)  A*V
* @ingroup AmMatrix
* @param  Output Vector ( should be already allocated )
* @param  Input 4x4 Matrix A
* @param  Input Vector 4x1 V
* @retval 0=failure,1=success
*/
int transform3DPointDVectorUsing4x4DMatrix(double * resultPoint3D, double * transformation4x4, double * point3D);


/**
* @brief Multiply a 4x4 matrix of floats with a float Vector (3D Point)  A*V
* @ingroup AmMatrix
* @param  Output Vector ( should be already allocated )
* @param  Input 4x4 Matrix A
* @param  Input Vector 4x1 V
* @retval 0=failure,1=success
*/
int transform3DPointFVectorUsing4x4FMatrix(float * resultPoint3D, float * transformation4x4, float * point3D);

/**
* @brief Multiply a the 3x3 rotational part of a 4x4 matrix with a Normal Vector (3D Point)  A*V
         Basically just performing a (3x3) x (3x1) operation from a (4x4) x (4x1) input
* @ingroup AmMatrix
* @param  Output Vector ( should be already allocated )
* @param  Input 4x4 Matrix A
* @param  Input Vector 4x1 V where W coordinate should be 0
* @retval 0=failure,1=success
*/
int transform3DNormalVectorUsing3x3DPartOf4x4DMatrix(double * resultPoint3D, double * transformation4x4, double * point3D);



int normalize3DPointFVector(float * vec);

/**
* @brief Normalize a 4x1 matrix with a Vector (3D Point)
* @ingroup AmMatrix
* @param  Input/Output Vector
* @retval 0=failure,1=success
*/
int normalize3DPointDVector(double * vec);

void doRPYTransformationF(
                         float *m,
                         float  rollInDegrees,
                         float  pitchInDegrees,
                         float  yawInDegrees
                        );

void doRPYTransformationD(
                         double *m,
                         double rollInDegrees,
                         double pitchInDegrees,
                         double yawInDegrees
                        );
/*
*/



/**
* @brief Produce a rotation and translation that will bring the scene to the coordinate frame of the camera in order to properly
         render objects..!

          This code produces the same matrix as the one produced by the following openGL calls

          glTranslatef(x,y,z);
          if ( roll!=0 )    { glRotatef(roll,0.0,0.0,1.0); }
          if ( heading!=0 ) { glRotatef(heading,0.0,1.0,0.0); }
          if ( pitch!=0 )   { glRotatef(pitch,1.0,0.0,0.0); }

          if ( (scaleX!=1.0) || (scaleY!=1.0) || (scaleZ!=1.0) )
                       {
                         glScalef( scaleX , scaleY , scaleZ );
                       }
* @ingroup AmMatrix
* @param  Input/Output Vector
* @retval 0=failure,1=success
*/
void create4x4DModelTransformation(
                                  double * m ,
                                  //Rotation Component
                                  double rotationX,//heading
                                  double rotationY,//pitch
                                  double rotationZ,//roll
                                  unsigned int rotationOrder,
                                  //Translation Component
                                  double x, double y, double z ,
                                  double scaleX, double scaleY, double scaleZ
                                 );




void create4x4FModelTransformation(
                                   float * m ,
                                  //Rotation Component
                                  float rotationX,//heading
                                  float rotationY,//pitch
                                  float rotationZ,//roll
                                  unsigned int rotationOrder,
                                  //Translation Component
                                  float x, float y, float z ,
                                  float scaleX, float scaleY, float scaleZ
                                 );




/**
* @brief Produce a rotation and translation that will bring the scene to the coordinate frame of the camera in order to properly
         render objects..!

          This code produces the same matrix as the one produced by the following openGL calls

          glLoadIdentity();
          glRotatef(rotationX_angleDegrees,-1.0,0,0);
          glRotatef(rotationY_angleDegrees,0,-1.0,0);
          glRotatef(rotationZ_angleDegrees,0,0,-1.0); }
          glTranslatef(-camera_pos_x, -camera_pos_y, -camera_pos_z);
* @ingroup AmMatrix
* @param  Input/Output Vector
* @retval 0=failure,1=success
*/
void create4x4DCameraModelViewMatrixForRendering(
                                                double * m ,
                                                //Rotation Component
                                                double rotationX_angleDegrees,
                                                double rotationY_angleDegrees,
                                                double rotationZ_angleDegrees ,
                                                //Translation Component
                                                double translationX_angleDegrees,
                                                double translationY_angleDegrees,
                                                double translationZ_angleDegrees
                                               );


#ifdef __cplusplus
}
#endif


#endif // MATRIX4X4TOOLS_H_INCLUDED
