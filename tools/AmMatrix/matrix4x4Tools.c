
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "matrix4x4Tools.h"

#define OPTIMIZED 1
//#define INTEL_OPTIMIZATIONS 0

#define PRINT_MATRIX_DEBUGGING 0

enum mat4x4Item
{
    I11 = 0 , I12 , I13 , I14 ,
    I21     , I22 , I23 , I24 ,
    I31     , I32 , I33 , I34 ,
    I41     , I42 , I43 , I44
};
 
enum mat4x4RTItem
{
    r0 = 0 , r1, r2 , t0 ,
    r3     , r4, r5 , t1 ,
    r6     , r7 ,r8 , t2 ,
    zero0  , zero1 , zero2 , one0
};
 
enum mat4x4EItem
{
    e0 = 0 , e1  , e2  , e3 ,
    e4     , e5  , e6  , e7 ,
    e8     , e9  , e10 , e11 ,
    e12    , e13 , e14 , e15
};

/* OUR MATRICES STORAGE
    0   1   2   3
    4   5   6   7
    8   9   10  11
    12  13  14  15
*/


void print4x4FMatrix(const char * str , float * matrix4x4,int forcePrint)
{
  #if PRINT_MATRIX_DEBUGGING
   forcePrint=1;
  #endif // PRINT_MATRIX_DEBUGGING

  if (forcePrint)
  {
  fprintf( stderr, " 4x4 float %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[0],matrix4x4[1],matrix4x4[2],matrix4x4[3]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[4],matrix4x4[5],matrix4x4[6],matrix4x4[7]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[8],matrix4x4[9],matrix4x4[10],matrix4x4[11]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f\n",matrix4x4[12],matrix4x4[13],matrix4x4[14],matrix4x4[15]);
  fprintf( stderr, "--------------------------------------\n");
  }
}

void print4x4DMatrix(const char * str , double * matrix4x4,int forcePrint)
{
  #if PRINT_MATRIX_DEBUGGING
   forcePrint=1;
  #endif // PRINT_MATRIX_DEBUGGING

  if (forcePrint)
  {
  fprintf( stderr, " 4x4 double %s \n",str);
  fprintf( stderr, "--------------------------------------\n");
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[0],matrix4x4[1],matrix4x4[2],matrix4x4[3]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[4],matrix4x4[5],matrix4x4[6],matrix4x4[7]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f,\n",matrix4x4[8],matrix4x4[9],matrix4x4[10],matrix4x4[11]);
  fprintf(stderr,"%0.4f,%0.4f,%0.4f,%0.4f\n",matrix4x4[12],matrix4x4[13],matrix4x4[14],matrix4x4[15]);
  fprintf( stderr, "--------------------------------------\n");
  }
}


void print4x4FMathematicaMatrix(const char * str , float * matrix3x3,int forcePrint)
{
  #if PRINT_MATRIX_DEBUGGING
    forcePrint=1;
  #endif // PRINT_MATRIX_DEBUGGING
  if (forcePrint)
  {
  fprintf( stderr, "%s = { { %f , %f , %f ,%f } , { %f , %f , %f , %f } , { %f , %f , %f , %f } , { %f , %f , %f , %f } }\n",str,
           matrix3x3[0],matrix3x3[1],matrix3x3[2],matrix3x3[3],
           matrix3x3[4],matrix3x3[5],matrix3x3[6],matrix3x3[7],
           matrix3x3[8],matrix3x3[9],matrix3x3[10],matrix3x3[11],
           matrix3x3[12],matrix3x3[13],matrix3x3[14],matrix3x3[15]);
  }
}



void copy3x3FMatrixTo4x4F(float * out,float * in)
{
  out[0]=in[0];   out[1]=in[1];   out[2]=in[2];   out[3]=0.0;
  out[4]=in[3];   out[5]=in[4];   out[6]=in[5];   out[7]=0.0;
  out[8]=in[6];   out[9]=in[7];   out[10]=in[8]; out[11]=0.0;
  out[12]=0.0; out[13]=0.0; out[14]=0.0; out[15]=1.0;
}

void copy4x4FMatrix(float * out,float * in)
{
  #if OPTIMIZED
  memcpy(out,in,16*sizeof(float));
  #else
  out[0]=in[0];   out[1]=in[1];   out[2]=in[2];   out[3]=in[3];
  out[4]=in[4];   out[5]=in[5];   out[6]=in[6];   out[7]=in[7];
  out[8]=in[8];   out[9]=in[9];   out[10]=in[10]; out[11]=in[11];
  out[12]=in[12]; out[13]=in[13]; out[14]=in[14]; out[15]=in[15];
  #endif // OPTIMIZED
}

void copy4x4DMatrix(double * out,double * in)
{
  #if OPTIMIZED
  memcpy(out,in,16*sizeof(double));
  #else
  out[0]=in[0];   out[1]=in[1];   out[2]=in[2];   out[3]=in[3];
  out[4]=in[4];   out[5]=in[5];   out[6]=in[6];   out[7]=in[7];
  out[8]=in[8];   out[9]=in[9];   out[10]=in[10]; out[11]=in[11];
  out[12]=in[12]; out[13]=in[13]; out[14]=in[14]; out[15]=in[15];
  #endif // OPTIMIZED
}


void copy4x4FMatrixTo4x4D(double * out,float * in)
{
  out[0]=(double)  in[0];   out[1]= (double) in[1];   out[2]= (double) in[2];  out[3]= (double) in[3];
  out[4]=(double)  in[4];   out[5]= (double) in[5];   out[6]= (double) in[6];  out[7]= (double) in[7];
  out[8]=(double)  in[8];   out[9]= (double) in[9];   out[10]=(double) in[10]; out[11]=(double) in[11];
  out[12]=(double) in[12];  out[13]=(double) in[13];  out[14]=(double) in[14]; out[15]=(double) in[15];
}

void copy4x4DMatrixTo4x4F(float * dest, double * m )
{
    dest[0]=(float)m[0];   dest[1]=(float)m[1];   dest[2]=(float)m[2];    dest[3]=(float)m[3];
    dest[4]=(float)m[4];   dest[5]=(float)m[5];   dest[6]=(float)m[6];    dest[7]=(float)m[7];
    dest[8]=(float)m[8];   dest[9]=(float)m[9];   dest[10]=(float)m[10];  dest[11]=(float)m[11];
    dest[12]=(float)m[12]; dest[13]=(float)m[13]; dest[14]=(float)m[14];  dest[15]=(float)m[15];
}

 

void create4x4FIdentityMatrix(struct Matrix4x4OfFloats * m)
{
  #if OPTIMIZED
   memset(m->m,0,16*sizeof(float));
   m->m[0] = 1.0;
   m->m[5] = 1.0;
   m->m[10] = 1.0;
   m->m[15] = 1.0;
  #else
    m->m[0] = 1.0;  m->m[1] = 0.0;  m->m[2] = 0.0;   m->m[3] = 0.0;
    m->m[4] = 0.0;  m->m[5] = 1.0;  m->m[6] = 0.0;   m->m[7] = 0.0;
    m->m[8] = 0.0;  m->m[9] = 0.0;  m->m[10] = 1.0;  m->m[11] =0.0;
    m->m[12]= 0.0;  m->m[13]= 0.0;  m->m[14] = 0.0;  m->m[15] = 1.0;
  #endif // OPTIMIZED
}

 
 
 

int floatPEq(float * element , float value )
{
 const float machineFloatPercision= 0.0001;
 if ( *element == value ) { return 1; }

 if ( ( value-machineFloatPercision<*element ) && (( *element<value+machineFloatPercision )) ) { return 1; }
 return 0;
}


int is4x4FIdentityMatrix(float  * m)
{
   return (
           (floatPEq(&m[0],1.0)) &&(floatPEq(&m[1],0.0)) &&(floatPEq(&m[2],0.0)) &&(floatPEq(&m[3],0.0)) &&
           (floatPEq(&m[4],0.0)) &&(floatPEq(&m[5],1.0)) &&(floatPEq(&m[6],0.0)) &&(floatPEq(&m[7],0.0)) &&
           (floatPEq(&m[8],0.0)) &&(floatPEq(&m[9],0.0)) &&(floatPEq(&m[10],1.0))&&(floatPEq(&m[11],0.0))&&
           (floatPEq(&m[12],0.0))&&(floatPEq(&m[13],0.0))&&(floatPEq(&m[14],0.0))&&(floatPEq(&m[15],1.0))
          );
}


int is4x4FIdentityMatrixS(struct Matrix4x4OfFloats * m)
{
   return is4x4FIdentityMatrix(m->m);
}


int is4x4FIdentityMatrixPercisionCompensating(struct Matrix4x4OfFloats  * m)
{
   return (
    (floatPEq(&m->m[0],1.0)) &&(floatPEq(&m->m[1],0.0)) &&(floatPEq(&m->m[2],0.0)) &&(floatPEq(&m->m[3],0.0)) &&
    (floatPEq(&m->m[4],0.0)) &&(floatPEq(&m->m[5],1.0)) &&(floatPEq(&m->m[6],0.0)) &&(floatPEq(&m->m[7],0.0)) &&
    (floatPEq(&m->m[8],0.0)) &&(floatPEq(&m->m[9],0.0)) &&(floatPEq(&m->m[10],1.0))&&(floatPEq(&m->m[11],0.0))&&
    (floatPEq(&m->m[12],0.0))&&(floatPEq(&m->m[13],0.0))&&(floatPEq(&m->m[14],0.0))&&(floatPEq(&m->m[15],1.0))
           );
}


void convert4x4FMatrixToRPY(struct Matrix4x4OfFloats * m ,float *roll,float *pitch,float *yaw)
{
 if (m->m[0] == 1.0f)
        {
          *yaw = atan2f( m->m[2], m->m[11]);
          *pitch = 0;
          *roll = 0;
        }
        else
 if (m->m[0] == -1.0f)
        {
          *yaw = atan2f(m->m[2], m->m[11]);
          *pitch = 0;
          *roll = 0;
        }
        else
        {
          *yaw = atan2(-m->m[8],m->m[0]);
          *pitch = asin(m->m[4]);
          *roll = atan2(-m->m[6],m->m[5]);
        } 
}




float degrees_to_radF(float degrees)
{
    return (float) degrees * ( (float) M_PI / 180.0 );
}
 

void create4x4FMatrixFromEulerAnglesXYZAllInOne(struct Matrix4x4OfFloats * m ,float eulX,float eulY,float eulZ)
{
    float x = degrees_to_radF(eulX);
    float y = degrees_to_radF(eulY);
    float z = degrees_to_radF(eulZ);

    float cr = cos( x );
    float sr = sin( x );
    float cp = cos( y );
    float sp = sin( y );
    float cy = cos( z );
    float sy = sin( z );

    m->m[0] = cp*cy ;
    m->m[1] = cp*sy;
    m->m[2] = -sp ;
    m->m[3] = 0; // 4x4


    float srsp = sr*sp;
    float crsp = cr*sp;

    m->m[4] = srsp*cy-cr*sy ;
    m->m[5] = srsp*sy+cr*cy ;
    m->m[6] = sr*cp ;
    m->m[7] = 0; // 4x4

    m->m[8] =  crsp*cy+sr*sy ;
    m->m[9] =  crsp*sy-sr*cy ;
    m->m[10]= cr*cp ;
    m->m[11]= 0; // 4x4

    // 4x4 last row
    m->m[12]= 0;
    m->m[13]= 0;
    m->m[14]= 0;
    m->m[15]= 1.0;
}


void create4x4FMatrixFromEulerAnglesZYX(struct Matrix4x4OfFloats * m ,float eulX,float eulY,float eulZ)
{
    //roll = X , pitch = Y , yaw = Z
    float x = degrees_to_radF(eulX);
    float y = degrees_to_radF(eulY);
    float z = degrees_to_radF(eulZ);


    float cr = cos(z);
    float sr = sin(z);
    float cp = cos(y);
    float sp = sin(y);
    float cy = cos(x);
    float sy = sin(x);

    float srsp = sr*sp;
    float crsp = cr*sp;

    m->m[0] = cr*cp;
    m->m[1] = crsp*sy - sr*cy;
    m->m[2] = crsp*cy + sr*sy;
    m->m[3] = 0;  // 4x4

    m->m[4] = sr*cp;
    m->m[5] = srsp*sy + cr*cy;
    m->m[6] = srsp*cy - cr*sy;
    m->m[7] = 0;  // 4x4

    m->m[8] = -sp;
    m->m[9] = cp*sy;
    m->m[10] = cp*cy;
    m->m[11] = 0;  // 4x4

     // 4x4 last row
    m->m[12] = 0;
    m->m[13] = 0;
    m->m[14] = 0;
    m->m[15] = 1;
}


//---------------------------------------------------------
void create4x4FRotationX(struct Matrix4x4OfFloats * m,float degrees)
{
    float radians = degrees_to_radF(degrees);

    create4x4FIdentityMatrix(m);

    float cosV = (float) cosf((float)radians);
    float sinV = (float) sinf((float)radians);

    // Rotate X formula.
    m->m[5] =    cosV; // [1,1]
    m->m[9] = -1*sinV; // [1,2]
    m->m[6] =    sinV; // [2,1]
    m->m[10] =   cosV; // [2,2]
}
//---------------------------------------------------------
void create4x4FRotationY(struct Matrix4x4OfFloats * m,float degrees)
{
    float radians = degrees_to_radF(degrees);

    create4x4FIdentityMatrix(m);

    float cosV = (float) cosf((float)radians);
    float sinV = (float) sinf((float)radians);

    // Rotate Y formula.
    m->m[0] =    cosV; // [0,0]
    m->m[2] = -1*sinV; // [2,0]
    m->m[8] =    sinV; // [0,2]
    m->m[10] =   cosV; // [2,2]
}
//---------------------------------------------------------
void create4x4FRotationZ(struct Matrix4x4OfFloats * m,float degrees)
{
    float radians = degrees_to_radF(degrees);

    create4x4FIdentityMatrix(m);

    float cosV = (float) cosf((float)radians);
    float sinV = (float) sinf((float)radians);

    // Rotate Z formula.
    m->m[0] =    cosV;  // [0,0]
    m->m[1] =    sinV;  // [1,0]
    m->m[4] = -1*sinV;  // [0,1]
    m->m[5] =    cosV;  // [1,1]
}
//---------------------------------------------------------
 




void create4x4FMatrixFromEulerAnglesWithRotationOrder(struct Matrix4x4OfFloats * m,float eulX, float eulY, float eulZ,unsigned int rotationOrder)
{
   //Initialize rotation matrix..
   create4x4FIdentityMatrix(m);

  if (rotationOrder==0)
  {
    //No rotation type, get's you back an Identity Matrix..
    fprintf(stderr,"create4x4MatrixFromEulerAnglesWithRotationOrder: No rotation order given, returning identity..\n");
    return;
  }
    float degreesX = eulX;
    float degreesY = eulY;
    float degreesZ = eulZ;
    struct Matrix4x4OfFloats rX={0};
    struct Matrix4x4OfFloats rY={0};
    struct Matrix4x4OfFloats rZ={0};

  //Assuming the rotation axis are correct
  //rX,rY,rZ should hold our rotation matrices
   create4x4FRotationX(&rX,degreesX);
   create4x4FRotationY(&rY,degreesY);
   create4x4FRotationZ(&rZ,degreesZ);

   #define SPEEDUP_ROTATION_CALCULATIONS 1

  switch (rotationOrder)
  {
    case ROTATION_ORDER_XYZ :
      #if SPEEDUP_ROTATION_CALCULATIONS 
       multiplyThree4x4FMatrices(m,&rX,&rY,&rZ);
      #else 
       multiplyTwo4x4FMatricesBuffered(m,m->m,rX.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rY.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rZ.m);
      #endif
    break;
    case ROTATION_ORDER_XZY :
      #if SPEEDUP_ROTATION_CALCULATIONS 
       multiplyThree4x4FMatrices(m,&rX,&rZ,&rY);
      #else 
       multiplyTwo4x4FMatricesBuffered(m,m->m,rX.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rZ.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rY.m);
      #endif
    break;
    case ROTATION_ORDER_YXZ :
      #if SPEEDUP_ROTATION_CALCULATIONS 
       multiplyThree4x4FMatrices(m,&rY,&rX,&rZ);
      #else 
       multiplyTwo4x4FMatricesBuffered(m,m->m,rY.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rX.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rZ.m);
      #endif
    break;
    case ROTATION_ORDER_YZX :
      #if SPEEDUP_ROTATION_CALCULATIONS 
       multiplyThree4x4FMatrices(m,&rY,&rZ,&rX);
      #else 
       multiplyTwo4x4FMatricesBuffered(m,m->m,rY.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rZ.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rX.m);
      #endif
    break;
    case ROTATION_ORDER_ZXY :
      #if SPEEDUP_ROTATION_CALCULATIONS 
       multiplyThree4x4FMatrices(m,&rZ,&rX,&rY);
      #else 
       multiplyTwo4x4FMatricesBuffered(m,m->m,rZ.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rX.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rY.m);
      #endif
    break;
    case ROTATION_ORDER_ZYX :
      #if SPEEDUP_ROTATION_CALCULATIONS 
       multiplyThree4x4FMatrices(m,&rZ,&rY,&rX);
      #else 
       multiplyTwo4x4FMatricesBuffered(m,m->m,rZ.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rY.m);
       multiplyTwo4x4FMatricesBuffered(m,m->m,rX.m);
      #endif
    break;
    case ROTATION_ORDER_RPY:
      fprintf(stderr,"create4x4MatrixFromEulerAnglesWithRotationOrderF can't handle RPY\n");
    break;
    default :
      fprintf(stderr,"create4x4MatrixFromEulerAnglesWithRotationOrderF: Error, Incorrect rotation type %u\n",rotationOrder);
    break;
  };
}



 
void create4x4FRotationMatrix(struct Matrix4x4OfFloats * m , float angle, float x, float y, float z)
{
    float const DEG2RAD=(float) M_PI/180;
    float c = cosf(angle * DEG2RAD);
    float s = sinf(angle * DEG2RAD);
    float xx = x * x;
    float xy = x * y;
    float xz = x * z;
    float yy = y * y;
    float yz = y * z;
    float zz = z * z;
    float one_min_c = (1 - c);
    float x_mul_s = x * s;
    float y_mul_s = y * s;
    float z_mul_s = z * s;

    m->m[0] = xx * one_min_c + c;
    m->m[1] = xy * one_min_c - z_mul_s;
    m->m[2] = xz * one_min_c + y_mul_s;
    m->m[3] = 0;
    m->m[4] = xy * one_min_c + z_mul_s;
    m->m[5] = yy * one_min_c + c;
    m->m[6] = yz * one_min_c - x_mul_s;
    m->m[7] = 0;
    m->m[8] = xz * one_min_c - y_mul_s;
    m->m[9] = yz * one_min_c + x_mul_s;
    m->m[10]= zz * one_min_c + c;
    m->m[11]= 0;
    m->m[12]= 0;
    m->m[13]= 0;
    m->m[14]= 0;
    m->m[15]= 1;
}

void create4x4FQuaternionMatrix(struct Matrix4x4OfFloats * m,float qX,float qY,float qZ,float qW)
{
    float yy2 = 2.0f * qY * qY;
    float xy2 = 2.0f * qX * qY;
    float xz2 = 2.0f * qX * qZ;
    float yz2 = 2.0f * qY * qZ;
    float zz2 = 2.0f * qZ * qZ;
    float wz2 = 2.0f * qW * qZ;
    float wy2 = 2.0f * qW * qY;
    float wx2 = 2.0f * qW * qX;
    float xx2 = 2.0f * qX * qX;
    m->m[0]  = - yy2 - zz2 + 1.0f;
    m->m[1]  = xy2 + wz2;
    m->m[2]  = xz2 - wy2;
    m->m[3]  = 0;
    m->m[4]  = xy2 - wz2;
    m->m[5]  = - xx2 - zz2 + 1.0f;
    m->m[6]  = yz2 + wx2;
    m->m[7]  = 0;
    m->m[8]  = xz2 + wy2;
    m->m[9]  = yz2 - wx2;
    m->m[10] = - xx2 - yy2 + 1.0f;
    m->m[11] = 0.0f;
    m->m[12] = 0.0;
    m->m[13] = 0.0;
    m->m[14] = 0.0;
    m->m[15] = 1.0f;
}


void create4x4FTranslationMatrix(struct Matrix4x4OfFloats * m , float x, float y, float z)
{
  #if OPTIMIZED
   memset(m->m,0,16*sizeof(float));
   m->m[0] = 1.0;
   m->m[3] = x;
   m->m[5] = 1.0;
   m->m[7] = y;
   m->m[10] = 1.0;
   m->m[11] = z;
   m->m[15] = 1.0;
  #else
    create4x4FIdentityMatrix(m);
    // Translate slots.
    m->m[3] = x; m->m[7] = y; m->m[11] = z;
  #endif // OPTIMIZED
}

 

void create4x4FScalingMatrix(struct Matrix4x4OfFloats * m, float scaleX, float scaleY, float scaleZ)
{
    if (m==0) { return ;} 
    create4x4FIdentityMatrix(m);
    // Scale slots.
    m->m[0] = scaleX; m->m[5] = scaleY; m->m[10] = scaleZ;
}
 
float det4x4FMatrix(float * mat)
{
 float * a = mat;

 float   detA  = a[I11] * a[I22] * a[I33]  * a[I44];
         detA += a[I11] * a[I23] * a[I34]  * a[I42];
         detA += a[I11] * a[I24] * a[I32]  * a[I43];

         detA += a[I12] * a[I21] * a[I34]  * a[I43];
         detA += a[I12] * a[I23] * a[I31]  * a[I44];
         detA += a[I12] * a[I24] * a[I33]  * a[I41];

         detA += a[I13] * a[I21] * a[I32]  * a[I44];
         detA += a[I13] * a[I22] * a[I34]  * a[I41];
         detA += a[I13] * a[I24] * a[I31]  * a[I42];

         detA += a[I14] * a[I21] * a[I33]  * a[I42];
         detA += a[I14] * a[I22] * a[I31]  * a[I43];
         detA += a[I14] * a[I23] * a[I32]  * a[I41];

  //FIRST PART DONE
         detA -= a[I11] * a[I22] * a[I34]  * a[I43];
         detA -= a[I11] * a[I23] * a[I32]  * a[I44];
         detA -= a[I11] * a[I24] * a[I33]  * a[I42];

         detA -= a[I12] * a[I21] * a[I33]  * a[I44];
         detA -= a[I12] * a[I23] * a[I34]  * a[I41];
         detA -= a[I12] * a[I24] * a[I31]  * a[I43];

         detA -= a[I13] * a[I21] * a[I34]  * a[I42];
         detA -= a[I13] * a[I22] * a[I31]  * a[I44];
         detA -= a[I13] * a[I24] * a[I32]  * a[I41];

         detA -= a[I14] * a[I21] * a[I32]  * a[I43];
         detA -= a[I14] * a[I22] * a[I33]  * a[I41];
         detA -= a[I14] * a[I23] * a[I31]  * a[I42];

 return detA;
}


int invert4x4FMatrix(struct Matrix4x4OfFloats * result,struct Matrix4x4OfFloats * mat)
{
 float * a = mat->m;
 float * b = result->m;
 float detA = det4x4FMatrix(mat->m);
 if (detA==0.0)
    {
      copy4x4FMatrix(result->m,mat->m);
      fprintf(stderr,"Matrix 4x4 cannot be inverted (det = 0)\n");
      return 0;
    }
 float one_div_detA = (float) 1 / detA;

 //FIRST LINE
 b[I11]  = a[I22] * a[I33] * a[I44] +  a[I23] * a[I34] * a[I42]  + a[I24] * a[I32] * a[I43];
 b[I11] -= a[I22] * a[I34] * a[I43] +  a[I23] * a[I32] * a[I44]  + a[I24] * a[I33] * a[I42];
 b[I11] *= one_div_detA;

 b[I12]  = a[I12] * a[I34] * a[I43] +  a[I13] * a[I32] * a[I44]  + a[I14] * a[I33] * a[I42];
 b[I12] -= a[I12] * a[I33] * a[I44] +  a[I13] * a[I34] * a[I42]  + a[I14] * a[I32] * a[I43];
 b[I12] *= one_div_detA;

 b[I13]  = a[I12] * a[I23] * a[I44] +  a[I13] * a[I24] * a[I42]  + a[I14] * a[I22] * a[I43];
 b[I13] -= a[I12] * a[I24] * a[I43] +  a[I13] * a[I22] * a[I44]  + a[I14] * a[I23] * a[I42];
 b[I13] *= one_div_detA;

 b[I14]  = a[I12] * a[I24] * a[I33] +  a[I13] * a[I22] * a[I34]  + a[I14] * a[I23] * a[I32];
 b[I14] -= a[I12] * a[I23] * a[I34] +  a[I13] * a[I24] * a[I32]  + a[I14] * a[I22] * a[I33];
 b[I14] *= one_div_detA;

 //SECOND LINE
 b[I21]  = a[I21] * a[I34] * a[I43] +  a[I23] * a[I31] * a[I44]  + a[I24] * a[I33] * a[I41];
 b[I21] -= a[I21] * a[I33] * a[I44] +  a[I23] * a[I34] * a[I41]  + a[I24] * a[I31] * a[I43];
 b[I21] *= one_div_detA;

 b[I22]  = a[I11] * a[I33] * a[I44] +  a[I13] * a[I34] * a[I41]  + a[I14] * a[I31] * a[I43];
 b[I22] -= a[I11] * a[I34] * a[I43] +  a[I13] * a[I31] * a[I44]  + a[I14] * a[I33] * a[I41];
 b[I22] *= one_div_detA;

 b[I23]  = a[I11] * a[I24] * a[I43] +  a[I13] * a[I21] * a[I44]  + a[I14] * a[I23] * a[I41];
 b[I23] -= a[I11] * a[I23] * a[I44] +  a[I13] * a[I24] * a[I41]  + a[I14] * a[I21] * a[I43];
 b[I23] *= one_div_detA;

 b[I24]  = a[I11] * a[I23] * a[I34] +  a[I13] * a[I24] * a[I31]  + a[I14] * a[I21] * a[I33];
 b[I24] -= a[I11] * a[I24] * a[I33] +  a[I13] * a[I21] * a[I34]  + a[I14] * a[I23] * a[I31];
 b[I24] *= one_div_detA;

 //THIRD LINE
 b[I31]  = a[I21] * a[I32] * a[I44] +  a[I22] * a[I34] * a[I41]  + a[I24] * a[I31] * a[I42];
 b[I31] -= a[I21] * a[I34] * a[I42] +  a[I22] * a[I31] * a[I44]  + a[I24] * a[I32] * a[I41];
 b[I31] *= one_div_detA;

 b[I32]  = a[I11] * a[I34] * a[I42] +  a[I12] * a[I31] * a[I44]  + a[I14] * a[I32] * a[I41];
 b[I32] -= a[I11] * a[I32] * a[I44] +  a[I12] * a[I34] * a[I41]  + a[I14] * a[I31] * a[I42];
 b[I32] *= one_div_detA;

 b[I33]  = a[I11] * a[I22] * a[I44] +  a[I12] * a[I24] * a[I41]  + a[I14] * a[I21] * a[I42];
 b[I33] -= a[I11] * a[I24] * a[I42] +  a[I12] * a[I21] * a[I44]  + a[I14] * a[I22] * a[I41];
 b[I33] *= one_div_detA;

 b[I34]  = a[I11] * a[I24] * a[I32] +  a[I12] * a[I21] * a[I34]  + a[I14] * a[I22] * a[I31];
 b[I34] -= a[I11] * a[I22] * a[I34] +  a[I12] * a[I24] * a[I31]  + a[I14] * a[I21] * a[I32];
 b[I34] *= one_div_detA;

 //FOURTH LINE
 b[I41]  = a[I21] * a[I33] * a[I42] +  a[I22] * a[I31] * a[I43]  + a[I23] * a[I32] * a[I41];
 b[I41] -= a[I21] * a[I32] * a[I43] +  a[I22] * a[I33] * a[I41]  + a[I23] * a[I31] * a[I42];
 b[I41] *= one_div_detA;

 b[I42]  = a[I11] * a[I32] * a[I43] +  a[I12] * a[I33] * a[I41]  + a[I13] * a[I31] * a[I42];
 b[I42] -= a[I11] * a[I33] * a[I42] +  a[I12] * a[I31] * a[I43]  + a[I13] * a[I32] * a[I41];
 b[I42] *= one_div_detA;

 b[I43]  = a[I11] * a[I23] * a[I42] +  a[I12] * a[I21] * a[I43]  + a[I13] * a[I22] * a[I41];
 b[I43] -= a[I11] * a[I22] * a[I43] +  a[I12] * a[I23] * a[I41]  + a[I13] * a[I21] * a[I42];
 b[I43] *= one_div_detA;

 b[I44]  = a[I11] * a[I22] * a[I33] +  a[I12] * a[I23] * a[I31]  + a[I13] * a[I21] * a[I32];
 b[I44] -= a[I11] * a[I23] * a[I32] +  a[I12] * a[I21] * a[I33]  + a[I13] * a[I22] * a[I31];
 b[I44] *= one_div_detA;


 #if PRINT_MATRIX_DEBUGGING
  print4x4FMatrix("Inverted Matrix From Source",a);
  print4x4FMatrix("Inverted Matrix To Target",b);
 #endif // PRINT_MATRIX_DEBUGGING

 return 1;
}


int transpose4x4FMatrix(float * mat)
{
  if (mat==0) { return 0; }
  /*       -------  TRANSPOSE ------->
      0   1   2   3           0  4  8   12
      4   5   6   7           1  5  9   13
      8   9   10  11          2  6  10  14
      12  13  14  15          3  7  11  15   */

  float tmp;
  tmp = mat[1]; mat[1]=mat[4];  mat[4]=tmp;
  tmp = mat[2]; mat[2]=mat[8];  mat[8]=tmp;
  tmp = mat[3]; mat[3]=mat[12]; mat[12]=tmp;


  tmp = mat[6]; mat[6]=mat[9]; mat[9]=tmp;
  tmp = mat[13]; mat[13]=mat[7]; mat[7]=tmp;
  tmp = mat[14]; mat[14]=mat[11]; mat[11]=tmp;

  return 1;
}

 
int transpose4x4DMatrix(double * mat)
{
  if (mat==0) { return 0; }
  /*       -------  TRANSPOSE ------->
      0   1   2   3           0  4  8   12
      4   5   6   7           1  5  9   13
      8   9   10  11          2  6  10  14
      12  13  14  15          3  7  11  15   */

  double tmp;
  tmp = mat[1]; mat[1]=mat[4];  mat[4]=tmp;
  tmp = mat[2]; mat[2]=mat[8];  mat[8]=tmp;
  tmp = mat[3]; mat[3]=mat[12]; mat[12]=tmp;


  tmp = mat[6]; mat[6]=mat[9]; mat[9]=tmp;
  tmp = mat[13]; mat[13]=mat[7]; mat[7]=tmp;
  tmp = mat[14]; mat[14]=mat[11]; mat[11]=tmp;

  return 1;
}


 
int multiplyTwo4x4DMatrices(double * result ,double * matrixA ,double * matrixB) 
{
  if ( (matrixA==0) || (matrixB==0) || (result==0) ) { return 0; }

  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"Multiplying 4x4 A and B \n");
  print4x4DMatrix("A", matrixA);
  print4x4DMatrix("B", matrixB);
  #endif // PRINT_MATRIX_DEBUGGING


  //MULTIPLICATION_RESULT FIRST ROW
  result[0]=matrixA[0] * matrixB[0] + matrixA[1] * matrixB[4]  + matrixA[2] * matrixB[8]  + matrixA[3] * matrixB[12];
  result[1]=matrixA[0] * matrixB[1] + matrixA[1] * matrixB[5]  + matrixA[2] * matrixB[9]  + matrixA[3] * matrixB[13];
  result[2]=matrixA[0] * matrixB[2] + matrixA[1] * matrixB[6]  + matrixA[2] * matrixB[10] + matrixA[3] * matrixB[14];
  result[3]=matrixA[0] * matrixB[3] + matrixA[1] * matrixB[7]  + matrixA[2] * matrixB[11] + matrixA[3] * matrixB[15];

  //MULTIPLICATION_RESULT SECOND ROW
  result[4]=matrixA[4] * matrixB[0] + matrixA[5] * matrixB[4]  + matrixA[6] * matrixB[8]  + matrixA[7] * matrixB[12];
  result[5]=matrixA[4] * matrixB[1] + matrixA[5] * matrixB[5]  + matrixA[6] * matrixB[9]  + matrixA[7] * matrixB[13];
  result[6]=matrixA[4] * matrixB[2] + matrixA[5] * matrixB[6]  + matrixA[6] * matrixB[10] + matrixA[7] * matrixB[14];
  result[7]=matrixA[4] * matrixB[3] + matrixA[5] * matrixB[7]  + matrixA[6] * matrixB[11] + matrixA[7] * matrixB[15];

  //MULTIPLICATION_RESULT THIRD ROW
  result[8] =matrixA[8] * matrixB[0] + matrixA[9] * matrixB[4]  + matrixA[10] * matrixB[8]   + matrixA[11] * matrixB[12];
  result[9] =matrixA[8] * matrixB[1] + matrixA[9] * matrixB[5]  + matrixA[10] * matrixB[9]   + matrixA[11] * matrixB[13];
  result[10]=matrixA[8] * matrixB[2] + matrixA[9] * matrixB[6]  + matrixA[10] * matrixB[10]  + matrixA[11] * matrixB[14];
  result[11]=matrixA[8] * matrixB[3] + matrixA[9] * matrixB[7]  + matrixA[10] * matrixB[11]  + matrixA[11] * matrixB[15];

  //MULTIPLICATION_RESULT FOURTH ROW
  result[12]=matrixA[12] * matrixB[0] + matrixA[13] * matrixB[4]  + matrixA[14] * matrixB[8]    + matrixA[15] * matrixB[12];
  result[13]=matrixA[12] * matrixB[1] + matrixA[13] * matrixB[5]  + matrixA[14] * matrixB[9]    + matrixA[15] * matrixB[13];
  result[14]=matrixA[12] * matrixB[2] + matrixA[13] * matrixB[6]  + matrixA[14] * matrixB[10]   + matrixA[15] * matrixB[14];
  result[15]=matrixA[12] * matrixB[3] + matrixA[13] * matrixB[7]  + matrixA[14] * matrixB[11]   + matrixA[15] * matrixB[15];

  #if PRINT_MATRIX_DEBUGGING
   print4x4DMatrix("AxB", result);
  #endif // PRINT_MATRIX_DEBUGGING

  return 1;
}


int multiplyThree4x4DMatrices(double * result , double * matrixA , double * matrixB , double * matrixC)
{
  if ( (matrixA==0) || (matrixB==0) || (matrixC==0) || (result==0) ) { return 0; }

  int i=0;
  double tmp[16];
  i+=multiplyTwo4x4DMatrices(tmp,matrixB,matrixC);
  i+=multiplyTwo4x4DMatrices(result , matrixA , tmp);

  return (i==2);
}



int multiplyTwo4x4FMatrices_Naive(float * result , float * matrixA , float * matrixB)
{
  if ( (matrixA==0) || (matrixB==0) || (result==0) ) { return 0; }

  #if PRINT_MATRIX_DEBUGGING
  fprintf(stderr,"Multiplying 4x4 A and B \n");
  print4x4FMatrix("A", matrixA);
  print4x4FMatrix("B", matrixB);
  #endif // PRINT_MATRIX_DEBUGGING


  //MULTIPLICATION_RESULT FIRST ROW
  result[0]=matrixA[0] * matrixB[0] + matrixA[1] * matrixB[4]  + matrixA[2] * matrixB[8]  + matrixA[3] * matrixB[12];
  result[1]=matrixA[0] * matrixB[1] + matrixA[1] * matrixB[5]  + matrixA[2] * matrixB[9]  + matrixA[3] * matrixB[13];
  result[2]=matrixA[0] * matrixB[2] + matrixA[1] * matrixB[6]  + matrixA[2] * matrixB[10] + matrixA[3] * matrixB[14];
  result[3]=matrixA[0] * matrixB[3] + matrixA[1] * matrixB[7]  + matrixA[2] * matrixB[11] + matrixA[3] * matrixB[15];

  //MULTIPLICATION_RESULT SECOND ROW
  result[4]=matrixA[4] * matrixB[0] + matrixA[5] * matrixB[4]  + matrixA[6] * matrixB[8]  + matrixA[7] * matrixB[12];
  result[5]=matrixA[4] * matrixB[1] + matrixA[5] * matrixB[5]  + matrixA[6] * matrixB[9]  + matrixA[7] * matrixB[13];
  result[6]=matrixA[4] * matrixB[2] + matrixA[5] * matrixB[6]  + matrixA[6] * matrixB[10] + matrixA[7] * matrixB[14];
  result[7]=matrixA[4] * matrixB[3] + matrixA[5] * matrixB[7]  + matrixA[6] * matrixB[11] + matrixA[7] * matrixB[15];

  //MULTIPLICATION_RESULT THIRD ROW
  result[8] =matrixA[8] * matrixB[0] + matrixA[9] * matrixB[4]  + matrixA[10] * matrixB[8]   + matrixA[11] * matrixB[12];
  result[9] =matrixA[8] * matrixB[1] + matrixA[9] * matrixB[5]  + matrixA[10] * matrixB[9]   + matrixA[11] * matrixB[13];
  result[10]=matrixA[8] * matrixB[2] + matrixA[9] * matrixB[6]  + matrixA[10] * matrixB[10]  + matrixA[11] * matrixB[14];
  result[11]=matrixA[8] * matrixB[3] + matrixA[9] * matrixB[7]  + matrixA[10] * matrixB[11]  + matrixA[11] * matrixB[15];

  //MULTIPLICATION_RESULT FOURTH ROW
  result[12]=matrixA[12] * matrixB[0] + matrixA[13] * matrixB[4]  + matrixA[14] * matrixB[8]    + matrixA[15] * matrixB[12];
  result[13]=matrixA[12] * matrixB[1] + matrixA[13] * matrixB[5]  + matrixA[14] * matrixB[9]    + matrixA[15] * matrixB[13];
  result[14]=matrixA[12] * matrixB[2] + matrixA[13] * matrixB[6]  + matrixA[14] * matrixB[10]   + matrixA[15] * matrixB[14];
  result[15]=matrixA[12] * matrixB[3] + matrixA[13] * matrixB[7]  + matrixA[14] * matrixB[11]   + matrixA[15] * matrixB[15];

  #if PRINT_MATRIX_DEBUGGING
   print4x4FMatrix("AxB", result);
  #endif // PRINT_MATRIX_DEBUGGING

  return 1;
}


#if INTEL_OPTIMIZATIONS
#include <xmmintrin.h>
#endif

int codeHasSSE()
{
#if INTEL_OPTIMIZATIONS
 return 1;
#endif // INTEL_OPTIMIZATIONS
return 0;
}

//__attribute__((aligned(16)))
int multiplyTwo4x4FMatrices_SSE(float * result , float * matrixA , float * matrixB)
{
#if INTEL_OPTIMIZATIONS
    float *A = matrixA;
    float *B = matrixB;
    float *C  = result;

    __m128 row1 = _mm_load_ps(&B[0]);
    __m128 row2 = _mm_load_ps(&B[4]);
    __m128 row3 = _mm_load_ps(&B[8]);
    __m128 row4 = _mm_load_ps(&B[12]);
    for(int i=0; i<4; i++) {
        __m128 brod1 = _mm_set1_ps(A[4*i + 0]);
        __m128 brod2 = _mm_set1_ps(A[4*i + 1]);
        __m128 brod3 = _mm_set1_ps(A[4*i + 2]);
        __m128 brod4 = _mm_set1_ps(A[4*i + 3]);
        __m128 row = _mm_add_ps(
                    _mm_add_ps(
                        _mm_mul_ps(brod1, row1),
                        _mm_mul_ps(brod2, row2)),
                    _mm_add_ps(
                        _mm_mul_ps(brod3, row3),
                        _mm_mul_ps(brod4, row4)));
        _mm_store_ps(&C[4*i], row);
    }
  return 1;
#else
  return multiplyTwo4x4FMatrices_Naive(result ,matrixA,matrixB);
#endif

}




int multiplyTwo4x4FMatricesS(struct Matrix4x4OfFloats * result ,struct Matrix4x4OfFloats * matrixA ,struct Matrix4x4OfFloats * matrixB)
{
#if INTEL_OPTIMIZATIONS
   return multiplyTwo4x4FMatrices_SSE(result->m,matrixA->m,matrixB->m);
#else 
   return multiplyTwo4x4FMatrices_Naive(result->m,matrixA->m,matrixB->m);
#endif
}




int multiplyTwo4x4FMatricesBuffered(struct Matrix4x4OfFloats * result , float * matrixA , float * matrixB)
{
  struct Matrix4x4OfFloats bufA;
   copy4x4FMatrix(bufA.m,matrixA);
  struct Matrix4x4OfFloats bufB;
   copy4x4FMatrix(bufB.m,matrixB);
  return  multiplyTwo4x4FMatricesS(result,&bufA,&bufB);
}


int multiplyThree4x4FMatrices(struct Matrix4x4OfFloats * result,struct Matrix4x4OfFloats * matrixA,struct Matrix4x4OfFloats * matrixB ,struct Matrix4x4OfFloats * matrixC)
{
  if ( (matrixA==0) || (matrixB==0) || (matrixC==0) || (result==0) ) { return 0; }

  int i=0;
  struct Matrix4x4OfFloats tmp;
  i+=multiplyTwo4x4FMatricesS(&tmp,matrixB,matrixC);
  i+=multiplyTwo4x4FMatricesS(result,matrixA,&tmp);

  return (i==2);
}


int multiplyFour4x4FMatrices(struct Matrix4x4OfFloats * result ,struct Matrix4x4OfFloats * matrixA ,struct Matrix4x4OfFloats * matrixB ,struct Matrix4x4OfFloats * matrixC ,struct Matrix4x4OfFloats * matrixD)
{
  if ( (matrixA==0) || (matrixB==0) || (matrixC==0) || (matrixD==0) || (result==0) ) { return 0; }

  int i=0;
  struct Matrix4x4OfFloats tmpA;
  struct Matrix4x4OfFloats tmpB;
  i+=multiplyTwo4x4FMatricesS(&tmpA,matrixC,matrixD);
  i+=multiplyTwo4x4FMatricesS(&tmpB,matrixB,&tmpA);
  i+=multiplyTwo4x4FMatricesS(result,matrixA,&tmpB);

  return (i==3);
}


int transform3DNormalVectorUsing3x3FPartOf4x4FMatrix(float * resultPoint3D,struct Matrix4x4OfFloats * transformation4x4,float * point3D)
{
  if ( (resultPoint3D==0) || (transformation4x4==0) || (point3D==0))  { return 0; }


  if (point3D[3]!=0.0)
  {
    fprintf(stderr,"Error with W coordinate transform3DNormalVectorUsing3x3FPartOf4x4FMatrix , should be zero  \n");
    return 0;
  }

  float * m = transformation4x4->m;
  register float X=point3D[0],Y=point3D[1],W=point3D[2];
  /*
  What we want to do ( in mathematica )
   { {me0,me1,me2} , {me3,me4,me5} , {me6,me7,me8} } * { { X } , { Y } , { W } }

   This gives us

  {
    {me2 W + me0 X + me1 Y},
    {me5 W + me3 X + me4 Y},
    {me8 W + me6 X + me7 Y}
  }
*/

  float * me0=&m[e0] , * me1=&m[e1] , * me2=&m[e2]  ;  //m[e3]  ignored
  float * me3=&m[e4] , * me4=&m[e5] , * me5=&m[e6]  ;  //m[e7]  ignored
  float * me6=&m[e8] , * me7=&m[e9] , * me8=&m[e10] ;  //m[e11] ignored
  //       last line ignored since we only want 3x3


  resultPoint3D[0] =  (*me2) * W + (*me0) * X + (*me1) * Y;
  resultPoint3D[1] =  (*me5) * W + (*me3) * X + (*me4) * Y;
  resultPoint3D[2] =  (*me8) * W + (*me6) * X + (*me7) * Y;
  resultPoint3D[3] =  0;

 // Ok we have our results but now to normalize our vector
  if(resultPoint3D[2]!=0.0)
  {
   resultPoint3D[0]/=resultPoint3D[2];
   resultPoint3D[1]/=resultPoint3D[2];
   resultPoint3D[2]=1.0; //resultPoint3D[2]/=resultPoint3D[2];
  }

  return 1;
}



 


int transform3DPointFVectorUsing4x4FMatrix(float * resultPoint3D,struct Matrix4x4OfFloats * transformation4x4, float * point3D)
{
  if ( (resultPoint3D==0) || (transformation4x4==0) || (point3D==0))  { return 0; }

/*
   What we want to do ( in mathematica )
   { {e0,e1,e2,e3} , {e4,e5,e6,e7} , {e8,e9,e10,e11} , {e12,e13,e14,e15} } * { { X } , { Y }  , { Z } , { W } }

   This gives us

  {
    {e3 W + e0 X + e1 Y + e2 Z},
    {e7 W + e4 X + e5 Y + e6 Z},
    {e11 W + e8 X + e9 Y + e10 Z},
    {e15 W + e12 X + e13 Y + e14 Z}
  }
*/
  float * m = transformation4x4->m;
  register float X=point3D[0],Y=point3D[1],Z=point3D[2],W=point3D[3];

  resultPoint3D[0] =  m[e3] * W + m[e0] * X + m[e1] * Y + m[e2] * Z;
  resultPoint3D[1] =  m[e7] * W + m[e4] * X + m[e5] * Y + m[e6] * Z;
  resultPoint3D[2] =  m[e11] * W + m[e8] * X + m[e9] * Y + m[e10] * Z;
  resultPoint3D[3] =  m[e15] * W + m[e12] * X + m[e13] * Y + m[e14] * Z;

  // Ok we have our results but now to normalize our vector
  if (resultPoint3D[3]!=0.0)
  {
   resultPoint3D[0]/=resultPoint3D[3];
   resultPoint3D[1]/=resultPoint3D[3];
   resultPoint3D[2]/=resultPoint3D[3];
   resultPoint3D[3]=1.0; // resultPoint3D[3]/=resultPoint3D[3];
   return 1;
  } else
  {
     fprintf(stderr,"Error with W coordinate after multiplication of 3D Point with 4x4 Matrix\n");
     fprintf(stderr,"Input Point was %0.2f %0.2f %0.2f %0.2f \n",point3D[0],point3D[1],point3D[2],point3D[3]);
     fprintf(stderr,"Output Point was %0.2f %0.2f %0.2f %0.2f \n",resultPoint3D[0],resultPoint3D[1],resultPoint3D[2],resultPoint3D[3]);
     return 0;
  }

 return 1;
}

int normalize3DPointFVector(float * vec)
{
  if ( vec[3]==1.0 ) { return 1; } else
  if ( vec[3]==0.0 )
  {
    fprintf(stderr,"normalize3DPointFVector cannot be normalized since element 3 is zero\n");
    return 0;
  }


  vec[0]=vec[0]/vec[3];
  vec[1]=vec[1]/vec[3];
  vec[2]=vec[2]/vec[3];
  vec[3]=1.0; // vec[3]=vec[3]/vec[3];

  return 1;
}

int normalize3DPointDVector(double * vec)
{
  if ( vec[3]==1.0 ) { return 1; } else
  if ( vec[3]==0.0 )
  {
    fprintf(stderr,"normalize3DPointDVector cannot be normalized since element 3 is zero\n");
    return 0;
  }


  vec[0]=vec[0]/vec[3];
  vec[1]=vec[1]/vec[3];
  vec[2]=vec[2]/vec[3];
  vec[3]=1.0; // vec[3]=vec[3]/vec[3];

  return 1;
}


void doRPYTransformationF(
                         struct Matrix4x4OfFloats * m,
                         float  rollInDegrees,
                         float  pitchInDegrees,
                         float  yawInDegrees
                        )
{
    struct Matrix4x4OfFloats intermediateMatrixPitch;
    struct Matrix4x4OfFloats intermediateMatrixHeading;
    struct Matrix4x4OfFloats intermediateMatrixRoll;
    create4x4FRotationMatrix(&intermediateMatrixRoll   , rollInDegrees,      0.0,   0.0,   1.0);
    create4x4FRotationMatrix(&intermediateMatrixHeading, yawInDegrees,       0.0,   1.0,   0.0);
    create4x4FRotationMatrix(&intermediateMatrixPitch  , pitchInDegrees,     1.0,   0.0,   0.0);

    multiplyThree4x4FMatrices(
                              m ,
                              &intermediateMatrixRoll ,
                              &intermediateMatrixHeading ,
                              &intermediateMatrixPitch
                            );
}

 


void create4x4FModelTransformation(
                                   struct Matrix4x4OfFloats * m ,
                                   //Rotation Component
                                   float rotationX,//heading
                                   float rotationY,//pitch
                                   float rotationZ,//roll
                                   unsigned int rotationOrder,
                                   //Translation Component
                                   float x, float y, float z ,
                                   float scaleX, float scaleY, float scaleZ
                                  )
{
   if (m==0) {return;}

    //fprintf(stderr,"Asked for a model transformation with RPY(%0.2f,%0.2f,%0.2f)",rollInDegrees,pitchInDegrees,yawInDegrees);
    //fprintf(stderr,"XYZ(%0.2f,%0.2f,%0.2f)",x,y,z);
    //fprintf(stderr,"scaled(%0.2f,%0.2f,%0.2f)\n",scaleX,scaleY,scaleZ);


    struct Matrix4x4OfFloats intermediateMatrixTranslation={0};
    create4x4FTranslationMatrix(
                                &intermediateMatrixTranslation,
                                x,
                                y,
                                z
                               );


    struct Matrix4x4OfFloats intermediateMatrixRotation;


    if ( (x==0) && (y==0) && (z==0) )
    {
      create4x4FIdentityMatrix(&intermediateMatrixRotation);
    } else
    if (rotationOrder>=ROTATION_ORDER_NUMBER_OF_NAMES)
    {
      fprintf(stderr,"create4x4FModelTransformation: wrong rotationOrder(%u)\n",rotationOrder);
    } else
    if (rotationOrder==ROTATION_ORDER_RPY)
    {
     //This is the old way to do this rotation
     doRPYTransformationF(
                          &intermediateMatrixRotation,
                          rotationZ,//roll,
                          rotationY,//pitch
                          rotationX//heading
                         );
    } else
    {
     //fprintf(stderr,"Using new model transform code\n");
     create4x4FMatrixFromEulerAnglesWithRotationOrder(
                                                      &intermediateMatrixRotation ,
                                                      rotationX,
                                                      rotationY,
                                                      rotationZ,
                                                      rotationOrder
                                                     );
    }


  if ( (scaleX!=1.0) || (scaleY!=1.0) || (scaleZ!=1.0) )
      {
        struct Matrix4x4OfFloats intermediateScalingMatrix;
        create4x4FScalingMatrix(&intermediateScalingMatrix,scaleX,scaleY,scaleZ);
        multiplyThree4x4FMatrices(m,&intermediateMatrixTranslation,&intermediateMatrixRotation,&intermediateScalingMatrix);
      } else
      {
         multiplyTwo4x4FMatricesS(m,&intermediateMatrixTranslation,&intermediateMatrixRotation);
      }
}



 

void create4x4FCameraModelViewMatrixForRendering(
                                                struct Matrix4x4OfFloats * m ,
                                                //Rotation Component
                                                float rotationX_angleDegrees,
                                                float rotationY_angleDegrees,
                                                float rotationZ_angleDegrees ,
                                                //Translation Component
                                                float translationX_angleDegrees,
                                                float translationY_angleDegrees,
                                                float translationZ_angleDegrees
                                               )
{
    if (m==0) {return;}

    struct Matrix4x4OfFloats intermediateMatrixRX;
    struct Matrix4x4OfFloats intermediateMatrixRY;
    struct Matrix4x4OfFloats intermediateMatrixRZ;
    create4x4FRotationMatrix(&intermediateMatrixRX, rotationX_angleDegrees,  -1.0,   0.0,   0.0);
    create4x4FRotationMatrix(&intermediateMatrixRY, rotationY_angleDegrees,   0.0,  -1.0,   0.0);
    create4x4FRotationMatrix(&intermediateMatrixRZ, rotationZ_angleDegrees,   0.0,   0.0,  -1.0);

    struct Matrix4x4OfFloats intermediateMatrixRotation;
    multiplyThree4x4FMatrices(
                              &intermediateMatrixRotation ,
                              &intermediateMatrixRX ,
                              &intermediateMatrixRY ,
                              &intermediateMatrixRZ
                             );

    struct Matrix4x4OfFloats intermediateMatrixTranslation;
    create4x4FTranslationMatrix(
                               &intermediateMatrixTranslation,
                               -translationX_angleDegrees,
                               -translationY_angleDegrees,
                               -translationZ_angleDegrees
                              );

    multiplyTwo4x4FMatricesS(m,&intermediateMatrixRotation,&intermediateMatrixTranslation);
}
