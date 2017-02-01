/** @file eulerToQuaternions.c
*   @brief  A Tool to convert euler angles to quaternions
*   @author Ammar Qammaz (AmmarkoV)
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <locale.h>

#include "../AmMatrix/quaternions.h"

int comma=0;

int forceUSLocaleToKeepOurSanity()
{
   setlocale(LC_ALL, "en_US.UTF-8");
   setlocale(LC_NUMERIC, "en_US.UTF-8");
   return 1;
}

int main(int argc, char *argv[])
{
    forceUSLocaleToKeepOurSanity();
    if (argc<3) { printf("EulerToQuaternions eulerAngleX eulerAngleY eulerAngleZ , you did not provide 3 arguments\n"); return 1; }

    unsigned int i=0;
    for (i=0; i<argc; i++)
    {
      if (strcmp("--comma",argv[i])==0) { comma=1; }
    }

    double euler[3];
    euler[0] = atof(argv[1]);
    euler[1] = atof(argv[2]);
    euler[2] = atof(argv[3]);

    double quaternions[4];
    euler2Quaternions(quaternions,euler,qXqYqZqW);

    if (comma) { printf("%f,%f,%f,%f\n",quaternions[0],quaternions[1],quaternions[2],quaternions[3]); } else
               { printf("%f %f %f %f\n",quaternions[0],quaternions[1],quaternions[2],quaternions[3]); }

    return 0;
}
