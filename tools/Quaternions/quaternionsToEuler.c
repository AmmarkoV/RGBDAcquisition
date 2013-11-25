#include <stdio.h>
#include <stdlib.h>

#include <math.h>
#include "quaternions.h"


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
