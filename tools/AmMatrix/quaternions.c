/** @file quaternions.c
*   @brief  A Library that provides quaternion functionality
*   @author Ammar Qammaz (AmmarkoV)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "quaternions.h"

//#define PI 3.141592653589793 precision anyone ? :P
#define PI 3.141592653589793238462643383279502884197
#define PI_DIV_180 0.01745329251
#define _180_DIV_PI 57.2957795131


#define USE_FAST_NORMALIZATION 0
#define USEATAN2 1
/* arctan and arcsin have a result between −π/2 and π/2. With three rotations between −π/2 and π/2 you can't have all possible orientations.
   We need to replace the arctan by atan2 to generate all the orientations. */


enum matrix4x4Enum
{
    m0_0=0,m0_1,m0_2,m0_3,
    m1_0,m1_1,m1_2,m1_3,
    m2_0,m2_1,m2_2,m2_3,
    m3_0,m3_1,m3_2,m3_3,
};

enum matrix3x3Enum
{
    m0=0,
    m1,m2,m3,m4,m5,m6,m7,m8
};


enum matrix3x3EnumTranspose
{
    mT0=0,mT3,mT6,
    mT1,mT4,mT7,
    mT2,mT5,mT8
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void handleQuaternionPackConvention(double qX,double qY,double qZ,double qW , double * packedQuaternionOutput,int quaternionConvention)
{
    switch (quaternionConvention)
    {
    case qWqXqYqZ  :
        packedQuaternionOutput[0]=qW;
        packedQuaternionOutput[1]=qX;
        packedQuaternionOutput[2]=qY;
        packedQuaternionOutput[3]=qZ;
        break;
    case qXqYqZqW :
        packedQuaternionOutput[0]=qX;
        packedQuaternionOutput[1]=qY;
        packedQuaternionOutput[2]=qZ;
        packedQuaternionOutput[3]=qW;
        break;

    default :
        fprintf(stderr,"Unhandled quaternion order given (%u) \n",quaternionConvention);
        break;
    }
}

void handleQuaternionUnpackConvention(double * packedQuaternionInput,double *qXOut,double *qYOut,double *qZOut,double *qWOut ,int quaternionConvention)
{
    switch (quaternionConvention)
    {
    case qWqXqYqZ  :
        *qWOut = packedQuaternionInput[0];
        *qXOut = packedQuaternionInput[1];
        *qYOut = packedQuaternionInput[2];
        *qZOut = packedQuaternionInput[3];
        break;
    case qXqYqZqW :
        *qXOut = packedQuaternionInput[0];
        *qYOut = packedQuaternionInput[1];
        *qZOut = packedQuaternionInput[2];
        *qWOut = packedQuaternionInput[3];
        break;

    default :
        fprintf(stderr,"Unhandled quaternion order given (%u) \n",quaternionConvention);
        break;
    }
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


int normalizeQuaternions(double *qX,double *qY,double *qZ,double *qW)
{
#if USE_FAST_NORMALIZATION
    // Works best when quat is already almost-normalized
    double f = (double) (3.0 - (((*qX) * (*qX)) + ( (*qY) * (*qY) ) + ( (*qZ) * (*qZ)) + ((*qW) * (*qW)))) / 2.0;
    *qX *= f;
    *qY *= f;
    *qZ *= f;
    *qW *= f;
#else
    double sqrtDown = (double) sqrt(((*qX) * (*qX)) + ( (*qY) * (*qY) ) + ( (*qZ) * (*qZ)) + ((*qW) * (*qW)));
    double f = (double) 1 / sqrtDown;
    *qX *= f;
    *qY *= f;
    *qZ *= f;
    *qW *= f;
#endif // USE_FAST_NORMALIZATION
    return 1;
}


double innerProductQuaternions(double qAX,double qAY,double qAZ,double qAW ,
                               double qBX,double qBY,double qBZ,double qBW)
{
    return (double) ((qAX * qBX) + (qAY * qBY)+ (qAZ * qBZ) + (qAW * qBW));
}


double anglesBetweenQuaternions(double qAX,double qAY,double qAZ,double qAW ,
                                double qBX,double qBY,double qBZ,double qBW)
{
    double rads= acos(innerProductQuaternions(qAX,qAY,qAZ,qAW,qBX,qBY,qBZ,qBW));

    return (double)  /*Why is the *2 needed ? */ 2* /*Why?*/ (rads * 180) / PI;
}


void multiplyQuaternions(double * qXOut,double * qYOut,double * qZOut,double * qWOut,
                         double qAX,double qAY,double qAZ,double qAW ,
                         double qBX,double qBY,double qBZ,double qBW)
{
    *qXOut = (qBX*qAX)-(qBY*qAY)-(qBZ*qAZ)-(qBW*qAW);
    *qYOut = (qBX*qAY)+(qBY*qAX)-(qBZ*qAW)+(qBW*qAZ);
    *qZOut = (qBX*qAZ)+(qBY*qAW)+(qBZ*qAX)-(qBW*qAY);
    *qWOut = (qBX*qAW)-(qBY*qAZ)+(qBZ*qAY)+(qBW*qAX);
}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


void euler2Quaternions(double * quaternions,double * euler,int quaternionConvention)
{
    //This conversion follows the rule euler X Y Z  to quaternions W X Y Z
    //Our input is degrees so we convert it to radians for the sin/cos functions
    double eX = (double) (euler[0] * PI) / 180;
    double eY = (double) (euler[1] * PI) / 180;
    double eZ = (double) (euler[2] * PI) / 180;

    //fprintf(stderr,"eX %f eY %f eZ %f\n",eX,eY,eZ);

    //http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    //eX Roll  φ - rotation about the X-axis
    //eY Pitch θ - rotation about the Y-axis
    //eZ Yaw   ψ - rotation about the Z-axis

    double cosX2 = cos((double) eX/2); //cos(φ/2);
    double sinX2 = sin((double) eX/2); //sin(φ/2);
    double cosY2 = cos((double) eY/2); //cos(θ/2);
    double sinY2 = sin((double) eY/2); //sin(θ/2);
    double cosZ2 = cos((double) eZ/2); //cos(ψ/2);
    double sinZ2 = sin((double) eZ/2); //sin(ψ/2);




    double qX = (sinX2 * cosY2 * cosZ2) - (cosX2 * sinY2 * sinZ2);
    double qY = (cosX2 * sinY2 * cosZ2) + (sinX2 * cosY2 * sinZ2);
    double qZ = (cosX2 * cosY2 * sinZ2) - (sinX2 * sinY2 * cosZ2);
    double qW = (cosX2 * cosY2 * cosZ2) + (sinX2 * sinY2 * sinZ2);

    handleQuaternionPackConvention(qX,qY,qZ,qW,quaternions,quaternionConvention);
}



void quaternions2Euler(double * euler,double * quaternions,int quaternionConvention)
{
    double qX,qY,qZ,qW;

    handleQuaternionUnpackConvention(quaternions,&qX,&qY,&qZ,&qW,quaternionConvention);

    //http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    //e1 Roll  - rX: rotation about the X-axis
    //e2 Pitch - rY: rotation about the Y-axis
    //e3 Yaw   - rZ: rotation about the Z-axis

    //Shorthand to go according to http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles
    double q0=qW , q1 = qX , q2 = qY , q3 = qZ;
    double q0q1 = (double) q0*q1 , q2q3 = (double) q2*q3;
    double q0q2 = (double) q0*q2 , q3q1 = (double) q3*q1;
    double q0q3 = (double) q0*q3 , q1q2 = (double) q1*q2;


    double eXDenominator = ( 1.0 - 2.0 * (q1*q1 + q2*q2) );
    if (eXDenominator == 0.0 )
    {
        fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n");
        return;
    }
    double eYDenominator = ( 1.0 - 2.0 * ( q2*q2 + q3*q3) );
    if (eYDenominator == 0.0 )
    {
        fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n");
        return;
    }


#if USEATAN2
    /* arctan and arcsin have a result between −π/2 and π/2. With three rotations between −π/2 and π/2 you can't have all possible orientations.
       We need to replace the arctan by atan2 to generate all the orientations. */
    /*eX*/ euler[0] = atan2( (2.0 *  (q0q1 + q2q3)) , eXDenominator ) ;
    /*eY*/ euler[1] = asin( 2.0 * (q0q2 - q3q1));
    /*eZ*/ euler[2] = atan2( (2.0 * (q0q3 + q1q2)) ,  eYDenominator );
#else
#warning "Please note that the compiled output does not generate all possible orientations"
#warning "See : http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles"
#warning "You are strongly suggested to #define USEATAN2 1 in quaternions.c "
    /*eX*/ euler[0] = atan( (2.0 *  (q0q1 + q2q3)) / eXDenominator) ;
    /*eY*/ euler[1] = asin( 2.0 * (q0q2 - q3q1));
    /*eZ*/ euler[2] = atan( (2.0 * (q0q3 + q1q2)) /  eYDenominator );
#endif // USEATAN2


    //Our output is in radians so we convert it to degrees for the user

    //Go from radians back to degrees
    euler[0] = (euler[0] * 180) / PI;
    euler[1] = (euler[1] * 180) / PI;
    euler[2] = (euler[2] * 180) / PI;

}


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void quaternionSlerp(double * qOut, double * q0,double * q1,double t)
{
    double product = (q0[0]*q1[0]) + (q0[1]*q1[1]) + (q0[2]*q1[2]) + (q0[3]*q1[3]); // -1,1)
    if (product<-1.0)
    {
        product=-1.0;
    }
    else if (product> 1.0)
    {
        product= 1.0;
    }

    double omega = acos(product);
    double absOmega = omega;
    if (absOmega<0.0)
    {
        absOmega=-1*absOmega;
    }
    if (absOmega < 1e-10)
    {
        if (omega<0.0)
        {
            omega = -1 * 1e-10;
        }
        else
        {
            omega = 1e-10;
        }
    }


    double som = sin(omega);
    double st0 = sin((1-t) * omega) / som;
    double st1 = sin(t * omega) / som;

    qOut[0] = q0[0]*st0 + q1[0]*st1;
    qOut[1] = q0[1]*st0 + q1[1]*st1;
    qOut[2] = q0[2]*st0 + q1[2]*st1;
    qOut[3] = q0[3]*st0 + q1[3]*st1;

    return;
}

void quaternion2Matrix3x3(double * matrix3x3,double * quaternions,int quaternionConvention)
{
    //http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    double qX,qY,qZ,qW;
    handleQuaternionUnpackConvention(quaternions,&qX,&qY,&qZ,&qW,quaternionConvention);

    double * m = matrix3x3;

    m[m0]=1 -(2*qY*qY) - (2*qZ*qZ); /*|*/  m[m1]=(2*qX*qY) - (2*qZ*qW);     /*|*/ m[m2]=(2*qX*qZ) + (2*qY*qW);
    m[m3]=(2*qX*qY) + (2*qZ*qW);    /*|*/  m[m4]=1 - (2*qX*qX) - (2*qZ*qZ); /*|*/ m[m5]=(2*qY*qZ) - (2*qX*qW);
    m[m6]=(2*qX*qZ) - (2*qY*qW);    /*|*/  m[m7]=(2*qY*qZ) + (2*qX*qW);     /*|*/ m[m8]=1 - (2*qX*qX) - (2*qY*qY);

    return ;
}


void Matrix4x42Quaternion(double * quaternions,int quaternionConvention,double * matrix4x4)
{
//http://www.gamasutra.com/view/feature/131686/rotating_objects_using_quaternions.php
    double qX,qY,qZ,qW;
    double m[4][4];
    m[0][0] = matrix4x4[m0_0];    m[0][1] = matrix4x4[m0_1];    m[0][2] = matrix4x4[m0_2];    m[0][3] = matrix4x4[m0_3];
    m[1][0] = matrix4x4[m1_0];    m[1][1] = matrix4x4[m1_1];    m[1][2] = matrix4x4[m1_2];    m[1][3] = matrix4x4[m1_3];
    m[2][0] = matrix4x4[m2_0];    m[0][1] = matrix4x4[m2_1];    m[0][2] = matrix4x4[m2_2];    m[0][3] = matrix4x4[m2_3];
    m[3][0] = matrix4x4[m3_0];    m[0][1] = matrix4x4[m3_1];    m[0][2] = matrix4x4[m3_2];    m[0][3] = matrix4x4[m3_3];

    float  tr, s, q[4];
    int    i, j, k;
    int nxt[3] = {1, 2, 0};
    tr = m[0][0] + m[1][1] + m[2][2];
// check the diagonal
    if (tr > 0.0)
    {
        s = sqrt (tr + 1.0);
        qW = s / 2.0;
        s = 0.5 / s;
        qX = (m[1][2] - m[2][1]) * s;
        qY = (m[2][0] - m[0][2]) * s;
        qZ = (m[0][1] - m[1][0]) * s;
    }
    else
    {
// diagonal is negative
        i = 0;
        if (m[1][1] > m[0][0]) i = 1;
        if (m[2][2] > m[i][i]) i = 2;
        j = nxt[i];
        k = nxt[j];
        s = sqrt ((m[i][i] - (m[j][j] + m[k][k])) + 1.0);
        q[i] = s * 0.5;
        if (s != 0.0) s = 0.5 / s;
        q[3] = (m[j][k] - m[k][j]) * s;
        q[j] = (m[i][j] + m[j][i]) * s;
        q[k] = (m[i][k] + m[k][i]) * s;
        qX = q[0];
        qY = q[1];
        qZ = q[2];
        qW = q[3];
    }

}


void axisAngle2Quaternion(double * quaternionOutput,double xx,double yy,double zz,double a, int quaternionConvention)
{
    // Here we calculate the sin( theta / 2) once for optimization
    double aDeg= a*PI_DIV_180;
    double result = sin( aDeg / 2.0 );

    // Calculate the x, y and z of the quaternion
    double x = xx * result;
    double y = yy * result;
    double z = zz * result;

    // Calcualte the w value by cos( theta / 2 )
    double w = cos( aDeg / 2.0 );
    normalizeQuaternions(&x,&y,&z,&w);

    handleQuaternionPackConvention(x,y,z,w,quaternionOutput,quaternionConvention);
}


void quaternionRotate(double * quaternion , double rotX , double rotY, double rotZ , double angleDegrees , int quaternionConvention)
{
    double rotationQuaternion[4]={0};
    axisAngle2Quaternion(rotationQuaternion,rotX,rotY,rotZ,angleDegrees,quaternionConvention);

    double result[4]={0};
    multiplyQuaternions(&result[pQX],&result[pQY],&result[pQZ],&result[pQW],
                        quaternion[pQX],quaternion[pQY],quaternion[pQZ],quaternion[pQW],
                        rotationQuaternion[pQX],rotationQuaternion[pQY],rotationQuaternion[pQZ],rotationQuaternion[pQW] );


   quaternion[0]=result[0]; quaternion[1]=result[1]; quaternion[2]=result[2]; quaternion[3]=result[3];
}




