#include "matrixCalculations.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "quaternions.h"
#include "matrixTools.h"
#include "matrix3x3Tools.h"
#include "matrix4x4Tools.h"
#include "solveLinearSystemGJ.h"
#include "solveHomography.h"

enum mat3x3MItem
{
    m0 = 0 , m1  , m2  ,
    m3     , m4  , m5  ,
    m6     , m7 ,  m8
};


enum mat4x4EItem
{
    e0 = 0 , e1  , e2  , e3 ,
    e4     , e5  , e6  , e7 ,
    e8     , e9  , e10 , e11 ,
    e12    , e13 , e14 , e15
};




/*
  TAKEN FROM http://www.lighthouse3d.com/opengl/maths/index.php?raytriint

*/

int rayIntersectsTriangle(float *p, float *d,float *v0, float *v1, float *v2)
{
	float e1[3],e2[3],h[3],s[3],q[3];
	float a,f,u,v;

	vector(e1,v1,v0);
	vector(e2,v2,v0);
	crossProduct(h,d,e2);
	a = innerProduct(e1,h);

	if (a > -0.00001 && a < 0.00001)
		return(0);

	f = 1/a;
	vector(s,p,v0);
	u = f * (innerProduct(s,h));

	if (u < 0.0 || u > 1.0)
		return(0);

	crossProduct(q,s,e1);
	v = f * innerProduct(d,q);
	if (v < 0.0 || u + v > 1.0)
		return(0);
	// at this stage we can compute t to find out where
	// the intersection point is on the line
	float t = f * innerProduct(e2,q);
	if (t > 0.00001) // ray intersection
		return(1);
	else // this means that there is a line intersection
		 // but not a ray intersection
		 return (0);
}


int rayIntersectsRectangle(float *p, float *d,float *v0, float *v1, float *v2, float *v3)
{
   if (  rayIntersectsTriangle(p,d,v0,v1,v2) )
     {
       return 1;
     }

   if (  rayIntersectsTriangle(p,d,v1,v2,v3) )
     {
       return 1;
     }

   return 0;
}

//http://ilab.usc.edu/wiki/index.php/Fast_Square_Root
inline float sqrt_fast_approximation(const float x)
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




double distanceBetween3DPoints(double * p1, double * p2)
{
  double x1 = p1[0] , y1 = p1[1] , z1 = p1[2];
  double x2 = p2[0] , y2 = p2[1] , z2 = p2[2];

  double dx=x1-x2;
  double dy=y1-y2;
  double dz=z1-z2;

  //I Could actually skip this
  //if (x1>=x2) { dx=x1-x2; } else { dx=x2-x1; }
  //if (y1>=y2) { dy=y1-y2; } else { dy=y2-y1; }
  //if (z1>=z2) { dz=z1-z2; } else { dz=z2-z1; }
  //==========================

  return (double) sqrt( (dx * dx) + (dy * dy) + (dz * dz) );
}


float distanceBetween3DPointsFast(float *x1,float*y1,float *z1,float *x2,float*y2,float *z2)
{
    //sqrt_fast_approximation
  float dx,dy,dz;

  if (*x1>=*x2) { dx=*x1-*x2; } else { dx=*x2-*x1; }
  if (*y1>=*y2) { dy=*y1-*y2; } else { dy=*y2-*y1; }
  if (*z1>=*z2) { dz=*z1-*z2; } else { dz=*z2-*z1; }

  return (float) sqrt_fast_approximation( (dx * dx) + (dy * dy) + (dz * dz) );
}

float squaredDistanceBetween3DPoints(float *x1,float*y1,float *z1,float *x2,float*y2,float *z2)
{
  float dx,dy,dz;

  if (*x1>=*x2) { dx=*x1-*x2; } else { dx=*x2-*x1; }
  if (*y1>=*y2) { dy=*y1-*y2; } else { dy=*y2-*y1; }
  if (*z1>=*z2) { dz=*z1-*z2; } else { dz=*z2-*z1; }

  return (float)  (dx * dx) + (dy * dy) + (dz * dz) ;
}





int convertRodriguezTo3x3(double * result,double * matrix)
{
  if ( (matrix==0) ||  (result==0) ) { return 0; }


  double x = matrix[0] , y = matrix[1] , z = matrix[2];
  double th = sqrt( x*x + y*y + z*z );
  double cosTh = cos(th);
  x = x / th; y = y / th; z = z / th;

  if ( th < 0.00001 )
    {
       create3x3IdentityMatrix(result);
       return 1;
    }

   //NORMAL RESULT
   result[0]=x*x * (1 - cosTh) + cosTh;          result[1]=x*y*(1 - cosTh) - z*sin(th);      result[2]=x*z*(1 - cosTh) + y*sin(th);
   result[3]=x*y*(1 - cosTh) + z*sin(th);        result[4]=y*y*(1 - cosTh) + cosTh;          result[5]=y*z*(1 - cosTh) - x*sin(th);
   result[6]=x*z*(1 - cosTh) - y*sin(th);        result[7]=y*z*(1 - cosTh) + x*sin(th);      result[8]=z*z*(1 - cosTh) + cosTh;

  #if PRINT_MATRIX_DEBUGGING
   fprintf(stderr,"rodriguez %f %f %f\n ",matrix[0],matrix[1],matrix[2]);
   print3x3DMatrix("Rodriguez Initial", result);
  #endif // PRINT_MATRIX_DEBUGGING

  return 1;
}


void changeYandZAxisOpenGL4x4Matrix(double * result,double * matrix)
{
  #if PRINT_MATRIX_DEBUGGING
   fprintf(stderr,"Invert Y and Z axis\n");
  #endif // PRINT_MATRIX_DEBUGGING

  double * invertOp = (double * ) malloc ( sizeof(double) * 16 );
  if (invertOp==0) { return; }

  create4x4IdentityMatrix(invertOp);
  invertOp[5]=-1;   invertOp[10]=-1;
  multiplyTwo4x4Matrices(result, matrix, invertOp);
  free(invertOp);
}



int projectPointsFrom3Dto2D(double * x2D, double * y2D , double * x3D, double *y3D , double * z3D , double * intrinsics , double * rotation3x3 , double * translation)
{
  double fx = intrinsics[0];
  double fy = intrinsics[4];
  double cx = intrinsics[2];
  double cy = intrinsics[5];

  double * t = translation;
  double * r = rotation3x3;

  //Result
  //fx * t0 + cx * t2 + (x3D) * ( fx * r0 + cx * r6 )  + (y3D) * ( fx * r1 + cx * r7 ) + (z3D) * (fx * r2 +cx * r8) / t3 + r7 x3D + r8 * y3D + r9 * z3D
  //fy * t1 + cy * t2 + x3D * ( fy * r3 + cy * r6 )  + y3D * ( fy * r4 + cy * r7 ) + z3D * (fy * r5 +cy * r8) / t3 + r7 x3D + r8 * y3D + r9 * z3D
  //1

  double x2DBuf =  fx * t[0] + cx * t[2] + (*x3D) * ( fx * r[0] + cx * r[6] )  + (*y3D) * ( fx * r[1] + cx * r[7] ) + (*z3D) * (fx * r[2] +cx * r[8]);
  double y2DBuf =  fy * t[1] + cy * t[2] + (*x3D) * ( fy * r[3] + cy * r[6] )  + (*y3D) * ( fy * r[4] + cy * r[7] ) + (*z3D) * (fy * r[5] +cy * r[8]);
  double scale =   t[2] + r[6] * (*x3D) + r[7] * (*y3D) + r[8] * (*z3D);

  if ( scale == 0.0 ) { fprintf(stderr,"could not projectPointsFrom3Dto2D"); return 0; }
  *x2D = x2DBuf / scale;
  *y2D = y2DBuf / scale;

 return 1;
}


int convertRodriguezAndTranslationTo4x4DUnprojectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit)
{
  double * matrix3x3Rotation = alloc4x4Matrix();    if (matrix3x3Rotation==0) { return 0; }

  //Our translation vector is ready to be used!
  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);
  #endif // PRINT_MATRIX_DEBUGGING

  //Our rodriguez vector should be first converted to a 3x3 Rotation matrix
  convertRodriguezTo3x3((double*) matrix3x3Rotation , rodriguez);

  //Shorthand variables for readable code :P
  double * m  = result4x4;
  double * rm = matrix3x3Rotation;
  double * tm = translation;


  //double scaleToDepthUnit = 1000.0; //Convert Unit to milimeters
  double Tx = tm[0]*scaleToDepthUnit;
  double Ty = tm[1]*scaleToDepthUnit;
  double Tz = tm[2]*scaleToDepthUnit;


  /*
      Here what we want to do is generate a 4x4 matrix that does the inverse transformation that our
      rodriguez and translation vector define

      In order to do that we should have the following be true

                                      (note the minus under)
      (   R  |  T  )       (   R trans |  - R trans * T  )         (   I  |  0   )
      (  --------- )    .  (  -------------------------- )     =   ( ----------- )
      (   0  |  1  )       (   0       |        1        )         (   0  |  I   )

      Using matlab to do the calculations we get the following matrix
  */

   m[0]=  rm[0];        m[1]= rm[3];        m[2]=  rm[6];       m[3]= -1.0 * ( rm[0]*Tx + rm[3]*Ty + rm[6]*Tz );
   m[4]=  rm[1];        m[5]= rm[4];        m[6]=  rm[7];       m[7]= -1.0 * ( rm[1]*Tx + rm[4]*Ty + rm[7]*Tz );
   m[8]=  rm[2];        m[9]= rm[5];        m[10]= rm[8];       m[11]=-1.0 * ( rm[2]*Tx + rm[5]*Ty + rm[8]*Tz );
   m[12]= 0.0;          m[13]= 0.0;         m[14]=0.0;          m[15]=1.0;


  print4x4DMatrix("ModelView", result4x4);
  free4x4Matrix(&matrix3x3Rotation);
  return 1;
}


int convertRodriguezAndTranslationToOpenGL4x4DProjectionMatrix(double * result4x4, double * rodriguez , double * translation , double scaleToDepthUnit )
{
  double * matrix3x3Rotation = alloc4x4Matrix();    if (matrix3x3Rotation==0) { return 0; }

  //Our translation vector is ready to be used!
  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"translation %f %f %f\n ",translation[0],translation[1],translation[2]);
  #endif // PRINT_MATRIX_DEBUGGING


  //Our rodriguez vector should be first converted to a 3x3 Rotation matrix
  convertRodriguezTo3x3((double*) matrix3x3Rotation , rodriguez);

  //Shorthand variables for readable code :P
  double * m  = result4x4;
  double * rm = matrix3x3Rotation;
  double * tm = translation;


  //double scaleToDepthUnit = 1000.0; //Convert Unit to milimeters
  double Tx = tm[0]*scaleToDepthUnit;
  double Ty = tm[1]*scaleToDepthUnit;
  double Tz = tm[2]*scaleToDepthUnit;

  /*
      Here what we want to do is generate a 4x4 matrix that does the normal transformation that our
      rodriguez and translation vector define
  */
   m[0]=  rm[0];        m[1]= rm[1];        m[2]=  rm[2];       m[3]= -Tx;
   m[4]=  rm[3];        m[5]= rm[4];        m[6]=  rm[5];       m[7]= -Ty;
   m[8]=  rm[6];        m[9]= rm[7];        m[10]= rm[8];       m[11]=-Tz;
   m[12]= 0.0;          m[13]= 0.0;         m[14]=0.0;          m[15]=1.0;


  #if PRINT_MATRIX_DEBUGGING
   print4x4DMatrix("ModelView", result4x4);
   fprintf(stderr,"Matrix will be transposed to become OpenGL format ( i.e. column major )\n");
  #endif // PRINT_MATRIX_DEBUGGING

  transpose4x4MatrixD(result4x4);

  free4x4Matrix(&matrix3x3Rotation);
  return 1;
}


int move3DPoint(double * resultPoint3D, double * transformation4x4, double * point3D  )
{
  return transform3DPointVectorUsing4x4Matrix(resultPoint3D,transformation4x4,point3D);
}



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
                                           )
{
   fprintf(stderr,"buildOpenGLProjectionForIntrinsics according to old Ammar code Image ( %u x %u )\n",imageWidth,imageHeight);
   fprintf(stderr,"fx %0.2f fy %0.2f , cx %0.2f , cy %0.2f , skew %0.2f \n",fx,fy,cx,cy,skew);
   fprintf(stderr,"Near %0.2f Far %0.2f \n",nearPlane,farPlane);


    // These parameters define the final viewport that is rendered into by
    // the camera.
    //     Left    Bottom   Right       Top
    double L = 0.0 , B = 0.0  , R = imageWidth , T = imageHeight;

    // near and far clipping planes, these only matter for the mapping from
    // world-space z-coordinate into the depth coordinate for OpenGL
    double N = nearPlane , F = farPlane;
    double R_sub_L = R-L , T_sub_B = T-B , F_sub_N = F-N , F_plus_N = F+N , F_mul_N = F*N;

    if  ( (R_sub_L==0) || (R_sub_L-1.0f==0) ||
          (T_sub_B==0) || (T_sub_B-1.0f==0) ||
          (F_sub_N==0) ) { fprintf(stderr,"Problem with image limits R-L=%f , T-B=%f , F-N=%f\n",R_sub_L,T_sub_B,F_sub_N); }


   // set the viewport parameters
   viewport[0] = L; viewport[1] = B; viewport[2] = R_sub_L; viewport[3] = T_sub_B;

   //OpenGL Projection Matrix ready for loading ( column-major ) , also axis compensated
   frustum[0] = -2.0f*fx/R_sub_L;     frustum[1] = 0.0f;                 frustum[2] = 0.0f;                              frustum[3] = 0.0f;
   frustum[4] = 0.0f;                 frustum[5] = 2.0f*fy/T_sub_B;      frustum[6] = 0.0f;                              frustum[7] = 0.0f;
   frustum[8] = 2.0f*cx/R_sub_L-1.0f; frustum[9] = 2.0f*cy/T_sub_B-1.0f; frustum[10]=-1.0*(F_plus_N/F_sub_N);            frustum[11] = -1.0f;
   frustum[12]= 0.0f;                 frustum[13]= 0.0f;                 frustum[14]=-2.0f*F_mul_N/(F_sub_N);            frustum[15] = 0.0f;
   //Matrix already in OpenGL column major format



   //TROUBLESHOOTING Left To Right Hand conventions , Thanks Damien 24-06-15
   double identMat[16];
   double finalFrutstrum[16];
   create4x4IdentityMatrix(identMat);
   identMat[10]=-1;
   multiplyTwo4x4Matrices(finalFrutstrum,identMat,frustum);
   copy4x4Matrix(frustum,finalFrutstrum);

}


float calculateDistance(float from_x,float from_y,float from_z,float to_x,float to_y,float to_z)
{
   float vect_x = from_x - to_x;
   float vect_y = from_y - to_y;
   float vect_z = from_z - to_z;

   return  (sqrt(pow(vect_x, 2) + pow(vect_y, 2) + pow(vect_z, 2)));

}


void vectorDirection(float src_x,float src_y,float src_z,float targ_x,float targ_y,float targ_z,float *vect_x,float *vect_y,float *vect_z)
{
    *vect_x = src_x - targ_x;
    *vect_y = src_y - targ_y;
    *vect_z = src_z - targ_z;

    float len = (sqrt(pow(*vect_x, 2) + pow(*vect_y, 2) + pow(*vect_z, 2)));
    if(len == 0) len = 1.0f;

    *vect_x /= len ;
    *vect_y /= len ;
    *vect_z /= len ;
}



void findNormal(float *v1x, float *v1y, float *v1z, float v2x, float v2y, float v2z, float v3x, float v3y, float v3z )
{ char x = 1;
  char y = 2;
  char z = 3;
  float temp_v1[3];
  float temp_v2[3];
  float temp_lenght;
  float CNormal[3];

temp_v1[x] = *v1x - v2x;
temp_v1[y] = *v1y - v2y;
temp_v1[z] = *v1z - v2z;

temp_v2[x] = v2x - v3x;
temp_v2[y] = v2y - v3y;
temp_v2[z] = v2z - v3z;

// calculate cross product
CNormal[x] = temp_v1[y]*temp_v2[z] - temp_v1[z]*temp_v2[y];
CNormal[y] = temp_v1[z]*temp_v2[x] - temp_v1[x]*temp_v2[z];
CNormal[z] = temp_v1[x]*temp_v2[y] - temp_v1[y]*temp_v2[x];

// normalize normal
temp_lenght =(CNormal[x]*CNormal[x])+ (CNormal[y]*CNormal[y])+ (CNormal[z]*CNormal[z]);

temp_lenght = sqrt(temp_lenght);

// prevent n/0
if (temp_lenght == 0) { temp_lenght = 1;}

CNormal[x] /= temp_lenght;
CNormal[y] /= temp_lenght;
CNormal[z] /= temp_lenght;


*v1x=CNormal[x];
*v1y=CNormal[y];
*v1z=CNormal[z];
}





int pointFromRelationWithObjectToAbsolute(double * absoluteOutPoint3DRotated, double * objectPosition , double * objectRotation3x3 ,  double * relativeInPoint3DUnrotated)
{
  //  What we want to do ( in mathematica )
  // (  { {r0,r1,r2,0} , {r3,r4,r5,0} , {r6,r7,r8,0} , {0,0,0,1} } * { { X }  , { Y }  , { Z } , { 1.0 } } ) + { {ObjX} , {ObjY} , {ObjZ} , { 0 }  }

  //We have a coordinate space in Relation to our object so we want to first rotate our point and then translate it
  //back to absolute coordinate space

  double objectRotation4x4[4*4]={0};
  //We make the 3x3 matrix onto a 4x4 by adding zeros and 1 as the diagonal element
  upscale3x3to4x4(objectRotation4x4,objectRotation3x3);

  objectRotation4x4[e3]=objectPosition[0];
  objectRotation4x4[e7]=objectPosition[1];
  objectRotation4x4[e11]=objectPosition[2];
  objectRotation4x4[e15]=1.0;

  transform3DPointVectorUsing4x4Matrix(absoluteOutPoint3DRotated,objectRotation4x4,relativeInPoint3DUnrotated);

  //Normalization is done automatically
  normalize3DPointVector(absoluteOutPoint3DRotated);

  return 1;
}


/*
    We have an object with an absolute Position X,Y,Z (objectPosition[]) and Rotation (objectRotation3x3[])
    We also have an absolute position of a 3D point , and we want to calculate the relative position
    of the 3D point in relation to the object ( unrotated relative position )
*/
int pointFromAbsoluteToInRelationWithObject(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation3x3 , double * absoluteInPoint3DRotated )
{
  //printf("pointFromAbsoluteToInRelationWithObject Using Inversion Code\n");
  double objectRotation4x4[4*4]={0};
  //We make the 3x3 matrix onto a 4x4 by adding zeros and 1 as the diagonal element
  upscale3x3to4x4(objectRotation4x4,objectRotation3x3);

  objectRotation4x4[e3]=objectPosition[0];
  objectRotation4x4[e7]=objectPosition[1];
  objectRotation4x4[e11]=objectPosition[2];
  objectRotation4x4[e15]=1.0;


  double objectInvRotation4x4[4*4]={0};
  invert4x4MatrixD(objectInvRotation4x4,objectRotation4x4);

  transform3DPointVectorUsing4x4Matrix(relativeOutPoint3DUnrotated,objectInvRotation4x4,absoluteInPoint3DRotated);
  return 1;
}







/*
    We have an object with an absolute Position X,Y,Z (objectPosition[]) and Rotation (objectRotation3x3[])
    We also have an absolute position of a 3D point , and we want to calculate the relative position
    of the 3D point in relation to the object ( unrotated relative position )
*/
int pointFromAbsoluteToRelationWithObject_PosXYZRotationXYZ(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectRotation , double * absoluteInPoint3DRotated )
{
    double objectRotation3x3[9];
    create3x3EulerRotationXYZOrthonormalMatrix(objectRotation3x3,objectRotation);

     pointFromAbsoluteToInRelationWithObject(relativeOutPoint3DUnrotated,objectPosition,objectRotation3x3,absoluteInPoint3DRotated);

    //We have to try to normalize the output point , although it should already be normalized..
    normalize3DPointVector(relativeOutPoint3DUnrotated);

    return 1;
}




/*
    We have an object with an absolute Position X,Y,Z (objectPosition[]) and Rotation (objectRotation3x3[])
    We also have an absolute position of a 3D point , and we want to calculate the relative position
    of the 3D point in relation to the object ( unrotated relative position )
*/
int pointFromAbsoluteToRelationWithObject_PosXYZQuaternionXYZW(double * relativeOutPoint3DUnrotated, double * objectPosition , double * objectQuaternion , double * absoluteInPoint3DRotated )
{
    double objectRotation3x3[9];

    //printf("Object Position is %f,%f,%f  \n", objectPosition[0], objectPosition[1], objectPosition[2] );
    //printf("Quaternion %f,%f,%f,%f \n",objectQuaternion[0],objectQuaternion[1],objectQuaternion[2],objectQuaternion[3]);
    normalizeQuaternions(&objectQuaternion[0],&objectQuaternion[1],&objectQuaternion[2],&objectQuaternion[3]);
    //printf("Normalized Quaternion %f,%f,%f,%f \n",objectQuaternion[0],objectQuaternion[1],objectQuaternion[2],objectQuaternion[3]);

    quaternion2Matrix3x3(objectRotation3x3,objectQuaternion,qXqYqZqW);

    //print3x3DMatrix("Quaternion to 3x3",objectRotation3x3);
    //print3x3DMathematicaMatrix("Quat3x3",objectRotation3x3);

    pointFromAbsoluteToInRelationWithObject(relativeOutPoint3DUnrotated,objectPosition,objectRotation3x3,absoluteInPoint3DRotated);

    //We have to try to normalize the output point , although it should already be normalized..
    normalize3DPointVector(relativeOutPoint3DUnrotated);

    return 1;
}


/*
    We have an object with a relative Position X,Y,Z to an Object (objectPosition[])
*/
int pointFromRelationWithObjectToAbsolute_PosXYZRotationXYZ(double * absoluteOutPoint3DRotated , double * objectPosition , double * objectRotation ,double * relativeInPoint3DUnrotated)
{
    double objectRotation3x3[9]={0};
    create3x3EulerRotationXYZOrthonormalMatrix(objectRotation3x3,objectRotation);
    pointFromRelationWithObjectToAbsolute(absoluteOutPoint3DRotated,objectPosition,objectRotation3x3,relativeInPoint3DUnrotated);

    //We have to try to normalize the output point , although it should already be normalized..
    normalize3DPointVector(absoluteOutPoint3DRotated);

    return 1;
}



/*
    We have an object with a relative Position X,Y,Z to an Object (objectPosition[])
*/
int pointFromRelationWithObjectToAbsolute_PosXYZQuaternionXYZW(double * absoluteOutPoint3DRotated , double * objectPosition , double * objectQuaternion ,double * relativeInPoint3DUnrotated)
{
    double objectRotation3x3[9];

    //printf("Object Position is %f,%f,%f  \n", objectPosition[0], objectPosition[1], objectPosition[2] );
    //printf("Quaternion %f,%f,%f,%f \n",objectQuaternion[0],objectQuaternion[1],objectQuaternion[2],objectQuaternion[3]);
    normalizeQuaternions(&objectQuaternion[0],&objectQuaternion[1],&objectQuaternion[2],&objectQuaternion[3]);
    //printf("Normalized Quaternion %f,%f,%f,%f \n",objectQuaternion[0],objectQuaternion[1],objectQuaternion[2],objectQuaternion[3]);

    quaternion2Matrix3x3(objectRotation3x3,objectQuaternion,qXqYqZqW);

    //print3x3DMatrix("Quaternion to 3x3",objectRotation3x3);
    //print3x3DMathematicaMatrix("Quat3x3",objectRotation3x3);

    pointFromRelationWithObjectToAbsolute(absoluteOutPoint3DRotated,objectPosition,objectRotation3x3,relativeInPoint3DUnrotated);

    //We have to try to normalize the output point , although it should already be normalized..
    normalize3DPointVector(absoluteOutPoint3DRotated);

    return 1;
}





void testMatrices()
{
   //testHomographySolver();
   testGJSolver();
  return ;


  double A[16]={ 1 ,2 ,3 ,4,
                 5 ,6 ,7 ,8,
                 9 ,10,11,12,
                 13,14,15,16
                };


  double B[16]={ 1 ,2 ,3 ,4,
                 4 ,3 ,2 ,1,
                 1 ,2 ,3 ,4,
                 4 ,3 ,2 ,1
                };

  double Res[16]={0};

  multiplyTwo4x4Matrices(Res,A,B);
/*
  28.000000 26.000000 24.000000 22.000000
  68.000000 66.000000 64.000000 62.000000
  108.000000 106.000000 104.000000 102.000000
  148.000000 146.000000 144.000000 142.000000*/

}
