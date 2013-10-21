#include <stdio.h>
#include <stdlib.h>

#include <math.h>

#define PI 3.14159265359

void quaternions2Euler(double * euler, double * quaternions)
{


  double q1x = quaternions[0];
  double q1y = quaternions[1];
  double q1z = quaternions[2];
  double q1w = quaternions[3];

  double e1 = euler[0];
  double e2 = euler[1];
  double e3 = euler[2];

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


  euler[0]=heading;
  euler[1]=attitude;
  euler[2]=bank;

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
