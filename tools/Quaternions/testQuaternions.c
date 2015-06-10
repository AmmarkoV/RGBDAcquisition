/** @file testQuaternions.c
*   @brief  A Tool to test quaternions library
*   @author Ammar Qammaz (AmmarkoV)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <locale.h>

#include "../AmMatrix/quaternions.h"

int forceUSLocaleToKeepOurSanity()
{
   setlocale(LC_ALL, "en_US.UTF-8");
   setlocale(LC_NUMERIC, "en_US.UTF-8");
   return 1;
}

int testRotation(double * quaternions , double * euler , double rX,double rY,double rZ)
{
    unsigned int i=0;
    signed int inc=10;
    for (i=0; i<5; i++)
    {
      quaternionRotate(quaternions,rX,rY,rZ, inc,qXqYqZqW);
      quaternions2Euler(euler,quaternions,qXqYqZqW);
      printf("Step %u %d  |  %f %f %f \n",i,inc,euler[0],euler[1],euler[2]);
    }

    inc=-10;
    for (i=0; i<5; i++)
    {
      quaternionRotate(quaternions,rX,rY,rZ, inc,qXqYqZqW);
      quaternions2Euler(euler,quaternions,qXqYqZqW);
      printf("Step %u %d | %f %f %f \n",i,inc,euler[0],euler[1],euler[2]);
    }

 return 1;
}


int main(int argc, char *argv[])
{
    forceUSLocaleToKeepOurSanity();
    if (argc<4) { printf("QuaternionsToEuler quatX quatY quatZ quatW, you did not provide 4 arguments\n"); return 1; }

    double euler[3]={0};
    euler[0] = atof(argv[1]);
    euler[1] = atof(argv[2]);
    euler[2] = atof(argv[3]);

    double quaternions[4];
    euler2Quaternions(quaternions,euler,qXqYqZqW);
    printf("Strarting from %f %f %f \n",euler[0],euler[1],euler[2]);

    testRotation(quaternions,euler,1.0,0.0,0.0);
    testRotation(quaternions,euler,0.0,1.0,0.0);
    testRotation(quaternions,euler,0.0,0.0,1.0);

    return 0;
}
