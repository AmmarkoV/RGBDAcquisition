#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#define PI 3.14159265359
#define _PIDiv180 (PI/180.0)
#define _180DivPI (180.0/PI)


enum quatOrder
{
  qWqXqYqZ=0,
  qXqYqZqW
};


/*
void quaternions2EulerNorm(double * euler, double * quaternions)
{
  double q1x = quaternions[0];
  double q1y = quaternions[1];
  double q1z = quaternions[2];
  double q1w = quaternions[3];

  double heading , attitude , bank;

	double test = q1x*q1y + q1z*q1w;
	if (test > 0.499) { // singularity at north pole
		heading = 2 * atan2(q1x,q1w);
		attitude = PI/2;
		bank = 0;
		return;
	}
	if (test < -0.499) { // singularity at south pole
		heading = -2 * atan2(q1x,q1w);
		attitude = - PI/2;
		bank = 0;
		return;
	}
    double sqx = q1x*q1x;
    double sqy = q1y*q1y;
    double sqz = q1z*q1z;
    heading = atan2(2*q1y*q1w-2*q1x*q1z , 1 - 2*sqy - 2*sqz);
	attitude = asin(2*test);
	bank = atan2(2*q1x*q1w-2*q1y*q1z , 1 - 2*sqx - 2*sqz);


    //Conversion to degrees
    bank *= _180DivPI;
    heading *= _180DivPI;
    attitude *= _180DivPI;

    //Bank = rotation about x axis
    euler[0]=bank;
    //Heading = rotation about y axis
    euler[1]=heading;
    //Attitude = rotation about z axis
    euler[2]=attitude ;

  return;
}

void quaternions2Euler(double * euler, double * quaternions)
{
    double q1x = quaternions[0];
    double q1y = quaternions[1];
    double q1z = quaternions[2];
    double q1w = quaternions[3];
    double heading , attitude , bank;

    double sqw = q1w*q1w;
    double sqx = q1x*q1x;
    double sqy = q1y*q1y;
    double sqz = q1z*q1z;
	double unit = sqx + sqy + sqz + sqw; // if normalised is one, otherwise is correction factor
	double test = q1x*q1y + q1z*q1w;
	if (test > 0.499*unit) { // singularity at north pole
		heading = 2 * atan2(q1x,q1w);
		attitude = PI/2;
		bank = 0;
		return;
	}
	if (test < -0.499*unit) { // singularity at south pole
		heading = -2 * atan2(q1x,q1w);
		attitude = -PI/2;
		bank = 0;
		return;
	}
    heading = atan2(2*q1y*q1w-2*q1x*q1z , sqx - sqy - sqz + sqw);
	attitude = asin(2*test/unit);
	bank = atan2(2*q1x*q1w-2*q1y*q1z , -sqx + sqy - sqz + sqw);



    //Conversion to degrees
    bank *= _180DivPI;
    heading *= _180DivPI;
    attitude *= _180DivPI;

    //Bank = rotation about x axis
    euler[0]=bank;
    //Heading = rotation about y axis
    euler[1]=heading;
    //Attitude = rotation about z axis
    euler[2]=attitude ;


  return;
}*/


void euler2Quaternions(double * quaternions,double * euler,int quaternionConvention)
{
  //This conversion follows the rule euler X Y Z  to quaternions W X Y Z
  double eX = euler[0];
  double eY = euler[1];
  double eZ = euler[2];

  //http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
  //eX Roll  φ - rotation about the X-axis
  //eY Pitch θ - rotation about the Y-axis
  //eZ Yaw   ψ - rotation about the Z-axis

  double cosX2 = cos(eX/2); //cos(φ/2);
  double sinX2 = sin(eX/2); //sin(φ/2);
  double cosY2 = cos(eY/2); //cos(θ/2);
  double sinY2 = sin(eY/2); //sin(θ/2);
  double cosZ2 = cos(eZ/2); //cos(ψ/2);
  double sinZ2 = sin(eZ/2); //sin(ψ/2);

  if ( quaternionConvention = qXqYqZqW )
  {
   /*qX*/ quaternions[0] = sinX2 * cosY2 * cosZ2 - cosX2 * sinY2 * sinZ2;
   /*qY*/ quaternions[1] = cosX2 * sinY2 * cosZ2 + sinX2 * cosY2 * sinZ2;
   /*qZ*/ quaternions[2] = cosX2 * cosY2 * cosZ2 - sinX2 * sinY2 * cosZ2;
   /*qW*/ quaternions[3] = cosX2 * cosY2 * cosZ2 + sinX2 * sinY2 * sinZ2;
  } else
  if (quaternionConvention = qWqXqYqZ )
  {
   /*qW*/ quaternions[0] = cosX2 * cosY2 * cosZ2 + sinX2 * sinY2 * sinZ2;
   /*qX*/ quaternions[1] = sinX2 * cosY2 * cosZ2 - cosX2 * sinY2 * sinZ2;
   /*qY*/ quaternions[2] = cosX2 * sinY2 * cosZ2 + sinX2 * cosY2 * sinZ2;
   /*qZ*/ quaternions[3] = cosX2 * cosY2 * cosZ2 - sinX2 * sinY2 * cosZ2;
  } else
  {
    fprintf(stderr,"Unhandled quaternion order given (%u) \n",quaternionConvention);
  }
}



void quaternions2Euler(double * quaternions,double * euler,int quaternionConvention)
{
    double qX,qY,qZ,qW;
    if (quaternionConvention = qWqXqYqZ )
    {
      qW = quaternions[0];
      qX = quaternions[1];
      qY = quaternions[2];
      qZ = quaternions[3];
    } else
    if (quaternionConvention = qXqYqZqW )
    {
      qX = quaternions[0];
      qY = quaternions[1];
      qZ = quaternions[2];
      qW = quaternions[3];
    }else
    {
     fprintf(stderr,"Unhandled quaternion order given (%u) \n",quaternionConvention);
    }

  //http://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
  //e1 Roll  - rX: rotation about the X-axis
  //e2 Pitch - rY: rotation about the Y-axis
  //e3 Yaw   - rZ: rotation about the Z-axis

  //Shorthand to go according to http://graphics.wikia.com/wiki/Conversion_between_quaternions_and_Euler_angles
  double q0=qW , q1 = qX , q2 = qY , q3 = qZ;
  double q0q1 = q0*q1 , q2q3 = q2*q3;
  double q0q2 = q0*q2 , q3q1 = q3*q1;
  double q0q3 = q0*q3 , q1q2 = q1*q2;


  double eXDenominator = ( 1.0 - 2 * (q1*q1 + q2*q2) );
  if (eXDenominator == 0.0 ) { fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n"); return; }
  double eYDenominator = ( 1.0 - 2 * ( q2*q2 + q3*q3) );
  if (eYDenominator == 0.0 ) { fprintf(stderr,"Gimbal lock detected , cannot convert to euler coordinates\n"); return; }


  /*eX*/ euler[0] = atan( (2 *  (q0q1 + q2q3)) / eXDenominator) ;
  /*eY*/ euler[1] = asin( 2 * (q0q2 - q3q1));
  /*eZ*/ euler[2] = atan( (2 * (q0q3 + q1q2)) /  eYDenominator );
}




int main(int argc, char *argv[])
{
    if (argc<4) { printf("QuaternionsToEuler quatA quatB quatC quatD, you did not provide 4 arguments\n"); return 1; }

    double euler[3]={0};

    double quaternions[4];
    quaternions[0] = atof(argv[1]);
    quaternions[1] = atof(argv[2]);
    quaternions[2] = atof(argv[3]);
    quaternions[3] = atof(argv[4]);

    quaternions2Euler(quaternions,euler,qXqYqZqW);


    printf("%f %f %f\n",euler[0],euler[1],euler[2]);

    return 0;
}
