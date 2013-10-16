#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359

void euler2Quaternions(double * euler, double * quaternions)
{
  double e1 = euler[0];
  double e2 = euler[1];
  double e3 = euler[2];

  double top , bottom ;

  double piDiv180=PI/180;
  //Calculating Quaternion 0
  quaternions[0] = sqrt(cos(e2*piDiv180)*cos(e1*piDiv180)+cos(e2*piDiv180)*cos(e3*piDiv180)-sin(e2*piDiv180)*sin(e1*piDiv180)*sin(e3*piDiv180)+cos(e1*piDiv180)* cos(e3*piDiv180)+1)/2;


  //Calculating Quaternion 1
  top = (cos(e1*piDiv180)*sin(e3*piDiv180)+cos(e2*piDiv180)*sin(e3*piDiv180)+sin(e2*piDiv180)*sin(e1*piDiv180)*cos(e3*piDiv180));
  bottom = sqrt(cos(e2*piDiv180)* cos(e1*piDiv180)+cos(e2*piDiv180)*cos(e3*piDiv180)-sin(e2*piDiv180)*sin(e1*piDiv180)*sin(e3*piDiv180)+cos(e1*piDiv180)*cos(e3*piDiv180)+1);
  if (bottom == 0.0 ) { fprintf(stderr,"Error , we get a division by zero for quaternion 1\nhttp://en.wikipedia.org/wiki/Gimbal_lock\n"); }
  quaternions[1] = (top/bottom)/2;


  //Calculating Quaternion 2
  top = (sin(e2*piDiv180)*sin(e3*piDiv180)-cos(e2*piDiv180)*sin(e1*piDiv180)*cos(e3*piDiv180)-sin(e1*piDiv180));
  bottom = sqrt(cos(e2*piDiv180)*cos(e1*piDiv180)+ cos(e2*piDiv180)*cos(e3*piDiv180)-sin(e2*piDiv180)*sin(e1*piDiv180)*sin(e3*piDiv180)+cos(e1*piDiv180)*cos(e3*piDiv180)+1);
  if (bottom == 0.0 ) { fprintf(stderr,"Error , we get a division by zero for quaternion 2\nhttp://en.wikipedia.org/wiki/Gimbal_lock\n"); }
  quaternions[2] = (top/bottom)/2;

  //Calculating Quaternion 2
  top = (sin(e2*piDiv180)*cos(e1*piDiv180)+sin(e2*piDiv180)*cos(e3*piDiv180)+cos(e2*piDiv180)*sin(e1*piDiv180)*sin(e3*piDiv180));
  bottom = sqrt(cos(e2*piDiv180)* cos(e1*piDiv180)+cos(e2*piDiv180)*cos(e3*piDiv180)-sin(e2*piDiv180)*sin(e1*piDiv180)*sin(e3*piDiv180)+cos(e1*piDiv180)*cos(e3*piDiv180)+1);
  quaternions[3] = (top/bottom)/2;
  if (bottom == 0.0 ) { fprintf(stderr,"Error , we get a division by zero for quaternion 2\nhttp://en.wikipedia.org/wiki/Gimbal_lock\n"); }

  return;
}










int main(int argc, char *argv[])
{
    if (argc<3) { printf("EulerToQuaternions eulerAngleA eulerAngleY eulerAngleZ , you did not provide 3 arguments\n"); return 1; }

    double euler[3];
    euler[0] = atof(argv[1]);
    euler[1] = atof(argv[2]);
    euler[2] = atof(argv[3]);

    double quaternions[4];
    euler2Quaternions(euler,quaternions);


    printf("%f %f %f %f\n",quaternions[0],quaternions[1],quaternions[2],quaternions[3]);

    return 0;
}
