/** @file quaternionsToEuler.c
*   @brief  A Tool to convert to quaternions to euler angles
*   @author Ammar Qammaz (AmmarkoV)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <locale.h>

#include "quaternions.h"

int forceUSLocaleToKeepOurSanity()
{
   setlocale(LC_ALL, "en_US.UTF-8");
   setlocale(LC_NUMERIC, "en_US.UTF-8");
   return 1;
}

int main(int argc, char *argv[])
{
    forceUSLocaleToKeepOurSanity();
    if (argc<4) { printf("QuaternionsToEuler quatX quatY quatZ quatW, you did not provide 4 arguments\n"); return 1; }

    double euler[3]={0};

    double quaternions[4];
    quaternions[0] = atof(argv[1]);
    quaternions[1] = atof(argv[2]);
    quaternions[2] = atof(argv[3]);
    quaternions[3] = atof(argv[4]);

    quaternions2Euler(euler,quaternions,qXqYqZqW);


    printf("%f %f %f\n",euler[0],euler[1],euler[2]);

    return 0;
}
