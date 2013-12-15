#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "quaternions.h"

//#define PI 3.141592653589793 precision anyone ? :P
#define PI 3.141592653589793238462643383279502884197

#define USE_FAST_NORMALIZATION 0
#define USEATAN2 1
  /* arctan and arcsin have a result between −π/2 and π/2. With three rotations between −π/2 and π/2 you can't have all possible orientations.
     We need to replace the arctan by atan2 to generate all the orientations. */

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

  switch (quaternionConvention )
  {
   case qXqYqZqW :
   /*qX*/ quaternions[0] = (sinX2 * cosY2 * cosZ2) - (cosX2 * sinY2 * sinZ2);
   /*qY*/ quaternions[1] = (cosX2 * sinY2 * cosZ2) + (sinX2 * cosY2 * sinZ2);
   /*qZ*/ quaternions[2] = (cosX2 * cosY2 * sinZ2) - (sinX2 * sinY2 * cosZ2);
   /*qW*/ quaternions[3] = (cosX2 * cosY2 * cosZ2) + (sinX2 * sinY2 * sinZ2);
   break;

   case qWqXqYqZ :
   /*qW*/ quaternions[0] = (cosX2 * cosY2 * cosZ2) + (sinX2 * sinY2 * sinZ2);
   /*qX*/ quaternions[1] = (sinX2 * cosY2 * cosZ2) - (cosX2 * sinY2 * sinZ2);
   /*qY*/ quaternions[2] = (cosX2 * sinY2 * cosZ2) + (sinX2 * cosY2 * sinZ2);
   /*qZ*/ quaternions[3] = (cosX2 * cosY2 * sinZ2) - (sinX2 * sinY2 * cosZ2);
   break;

   default :
    fprintf(stderr,"Unhandled quaternion order given (%u) \n",quaternionConvention);
   break;
  };

}



void quaternions2Euler(double * euler,double * quaternions,int quaternionConvention)
{
    double qX,qY,qZ,qW;

    switch (quaternionConvention)
     {
       case qWqXqYqZ  :
       qW = quaternions[0];
       qX = quaternions[1];
       qY = quaternions[2];
       qZ = quaternions[3];
       break;

       case qXqYqZqW :
       qX = quaternions[0];
       qY = quaternions[1];
       qZ = quaternions[2];
       qW = quaternions[3];
       break;

       default :
       fprintf(stderr,"Unhandled quaternion order given (%u) \n",quaternionConvention);
       break;
     }

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
  if (eXDenominator == 0.0 ) { fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n"); return; }
  double eYDenominator = ( 1.0 - 2.0 * ( q2*q2 + q3*q3) );
  if (eYDenominator == 0.0 ) { fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n"); return; }


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
