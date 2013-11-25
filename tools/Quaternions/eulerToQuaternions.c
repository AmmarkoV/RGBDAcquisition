#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "quaternions.h"

int main(int argc, char *argv[])
{
    if (argc<3) { printf("EulerToQuaternions eulerAngleX eulerAngleY eulerAngleZ , you did not provide 3 arguments\n"); return 1; }

    double euler[3];
    euler[0] = atof(argv[1]);
    euler[1] = atof(argv[2]);
    euler[2] = atof(argv[3]);

    double quaternions[4];
    euler2Quaternions(quaternions,euler,qXqYqZqW);

    printf("%f %f %f %f\n",quaternions[0],quaternions[1],quaternions[2],quaternions[3]);

    return 0;
}
