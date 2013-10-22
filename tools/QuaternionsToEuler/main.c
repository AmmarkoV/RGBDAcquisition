#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#define PI 3.14159265359
#define _PIDiv180 (PI/180.0)
#define _180DivPI (180.0/PI)

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

    quaternions2Euler(euler,quaternions);


    printf("%f %f %f\n",euler[0],euler[1],euler[2]);

    return 0;
}
